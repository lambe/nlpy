"""
 SBMIN
 Trust-Region Method for box constrained Programming.

 A first unconstrained optimization solver in Python
"""

from nlpy.optimize.solvers import lbfgs    # For Hessian Approximate
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.tools import norms
from nlpy.tools.timing import cputime
from nlpy.tools.exceptions import UserExitRequest
import numpy
import logging
from math import sqrt
from nlpy.model import NLPModel


__docformat__ = 'restructuredtext'


class SBMINFramework:
    """
    An abstract framework for a trust-region-based algorithm for nonlinear
    box constrained programming. Instantiate using

    `SBMIN = SBMINFramework(nlp, TR, TrSolver)`

    :parameters:

        :nlp:   a :class:`NLPModel` object representing the problem. For
                instance, nlp may arise from an AMPL model
        :TR:    a :class:`TrustRegionFramework` object
        :TrSolver:   a :class:`TrustRegionSolver` object.


    :keywords:

        :x0:           starting point                    (default nlp.x0)
        :reltol:       absolute stopping tolerance       (default 1.0e-6)
        :maxiter:      maximum number of iterations      (default max(1000,10n))
        :logger_name:  name of a logger object that can be used in the post
                       iteration                         (default None)
        :verbose:      print some info if True                 (default True)

    Once a `SBMINFramework` object has been instantiated and the problem is
    set up, solve problem by issuing a call to `SBMIN.solve()`. The algorithm
    stops as soon as the infinity norm of the projected gradient into the
    feasible box falls below `reltol`

    """

    def __init__(self, nlp, TR, TrSolver, **kwargs):

        self.nlp    = nlp
        self.TR     = TR
        self.TrSolver = TrSolver
        self.solver   = None    # Will point to solver data in Solve()
        self.iter   = 0         # Iteration counter
        self.x      = kwargs.get('x0', self.nlp.x0.copy())
        self.f      = None
        self.f0     = None
        self.g      = None
        self.g_old  = None
        self.save_g = False              # For methods that need g_{k-1} and g_k
        self.gnorm  = None
        self.g0     = None
        self.tsolve = 0.0

        self.reltol  = kwargs.get('reltol', 1.0e-5)
        self.maxiter = kwargs.get('maxiter', max(1000, 10*self.nlp.n))
        self.verbose = kwargs.get('verbose', True)
        self.logger = kwargs.get('logger', None)
        self.total_bqpiter = 0

        self.hformat = '%-5s  %8s  %7s  %5s  %8s  %8s  %4s'
        self.header  = self.hformat % ('     Iter','f(x)','|g(x)|','cg','rho','Radius','Stat')
        self.hlen   = len(self.header)
        self.hline  = '     '+'-' * self.hlen
        self.format = '     %-5d  %8.2e  %7.1e  %5d  %8.1e  %8.1e  %4s'
        self.format0= '     %-5d  %8.2e  %7.1e  %5s  %8s  %8s  %4s'
        self.radii = None

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlpy.sbmin')
        self.log = logging.getLogger(logger_name)
        if not self.verbose:
            self.log.propagate=False

    def hprod(self, v, **kwargs):
        """
        Default hprod based on nlp's hprod. User should overload to
        provide a custom routine, e.g., a quasi-Newton approximation.
        """
        return self.nlp.hprod(self.x, self.nlp.pi0, v)

    def projected_gradient(self, x, g):
        """
        Compute the projected gradient of f(x) into the feasible box

                   l <= x <= u

        """
        p = x - g
        med = numpy.maximum(numpy.minimum(p,self.nlp.Uvar),self.nlp.Lvar)
        q = x - med

        return q

    def PostIteration(self, **kwargs):
        """
        Override this method to perform work at the end of an iteration. For
        example, use this method for updating a LBFGS Hessian
        """
        return None

    def Solve(self, **kwargs):

        nlp = self.nlp

        # Gather initial information.
        self.f      = self.nlp.obj(self.x)
        self.f0     = self.f
        self.g      = self.nlp.grad(self.x)  # Current  gradient
        self.g_old  = self.g                 # Previous gradient
        self.pgnorm = numpy.max(numpy.abs( \
                                self.projected_gradient(self.x,self.g)))
        self.pg0 = self.pgnorm

        # Reset initial trust-region radius.
        self.TR.Delta = .2#0.1 * norms.norm2(self.g)

        self.radii = [ self.TR.Delta ]

        reltol = self.reltol
        step_status = None
        exitIter = exitUser = False
        exitOptimal = self.pgnorm <= reltol
        status = ''

        t = cputime()

        # Print out header and initial log.
        if self.verbose:
            print self.hline
            print self.header
            print self.hline
            print self.format0 % (self.iter, self.f,
                                             self.pgnorm, '', '',
                                             '', '')

        while not (exitUser or exitOptimal or exitIter):

            self.iter += 1

            # Save current gradient for LBFGS needs
            if self.save_g:
                self.g_old = self.g.copy()

            # Iteratively minimize the quadratic model in the trust region
            #          m(d) = <g, d> + 1/2 <d, Hd>
            #     s.t.     ll <= d <= uu

            qp = TrustBQPModel(self.nlp, self.x, self.TR.Delta)

            self.solver = self.TrSolver(qp, qp.obj)
            self.solver.Solve(reltol=reltol)

            step = self.solver.step
            stepnorm = self.solver.stepNorm
            bqpiter = self.solver.niter

            # Obtain model value at next candidate
            m = self.solver.m
            if m is None:
                m = numpy.dot(self.g, step) + 0.5*numpy.dot(step, H * step)

            self.total_bqpiter += bqpiter
            x_trial = self.x + step
            f_trial = nlp.obj(x_trial)

            rho  = self.TR.Rho(self.f, f_trial, m)
            step_status = 'Rej'

            if rho >= self.TR.eta1:

                # Trust-region step is succesfull.
                self.TR.UpdateRadius(rho, stepnorm)
                self.x = x_trial
                self.f = f_trial
                self.g = nlp.grad(self.x)
                self.pgnorm = numpy.max(numpy.abs( \
                                        self.projected_gradient(self.x,self.g)))

                step_status = 'Acc'

            else:
                # Trust-region step is unsuccessfull.
                self.TR.UpdateRadius(rho, stepnorm)

            self.step_status = step_status
            self.radii.append(self.TR.Delta)
            status = ''
            try:
                self.PostIteration()
            except UserExitRequest:
                status = 'usr'

            # Print out header, say, every 20 iterations
            if self.verbose:
                self.log.info(self.hline)
                self.log.info(self.header)
                self.log.info(self.hline)

            if self.verbose:
                pstatus = step_status if step_status != 'Acc' else ''
                print self.format % (self.iter, self.f,
                          self.pgnorm, bqpiter, rho,
                          self.radii[-2], pstatus)

            exitOptimal = self.pgnorm <= reltol
            exitIter    = self.iter > self.maxiter
            exitUser    = status == 'usr'

        self.tsolve = cputime() - t    # Solve time

        # Set final solver status.
        if status == 'usr':
            pass
        elif self.pgnorm <= reltol:
            status = 'opt'
        else: # self.iter > self.maxiter:
            status = 'itr'
        self.status = status


class SBMINLbfgsFramework(SBMINFramework):
    """
    Class SBMINLbfgsFramework is a subclass of SBMINFramework. The method is
    based on a trust-region-based algorithm for nonlinear box constrained
    programming.
    The only difference is that a limited-memory BFGS Hessian approximation
    is used and maintained along the iterations. See class SBMINFramework for
    more information.
    """

    def __init__(self, nlp, TR, TrSolver, **kwargs):

        SBMINFramework.__init__(self, nlp, TR, TrSolver, **kwargs)
        self.save_g = True

    def PostIteration(self, **kwargs):
        """
        This method updates the limited-memory BFGS Hessian by appending
        the most rencet (s,y) pair to it and possibly discarding the oldest one
        if all the memory has been used.
        """
        s = self.solver.step
        y = self.g - self.g_old
        self.nlp.hupdate(s, y)




class TrustBQPModel(NLPModel):
    """
    Class for defining a Model to pass to BQP solver:
                min     m(x) = <g, x-x_k> + 1/2 < x-x_k, H*(x-x_k)>
                s.t.       l <= x <= u
                           || x - x_k ||_oo  <= delta

    where `g` is the gradient evaluated at x_k
    and `H` is  the Hessian evaluated at x_k.
    """

    def __init__(self, nlp, x_k, delta, **kwargs):

        Delta = delta*numpy.ones(nlp.n)
        Lvar = numpy.maximum(nlp.Lvar - x_k, -Delta)
        Uvar = numpy.minimum(nlp.Uvar - x_k, Delta)

        NLPModel.__init__(self, n=nlp.n, m=nlp.m, name='TrustRegionSubproblem',
                          Lvar=Lvar ,Uvar =Uvar)
        self.nlp = nlp
        self.x0 = numpy.zeros(self.nlp.n)
        self.x_k = x_k
        self.delta = delta
        self.g_k = self.nlp.grad(self.x_k)

        # Saved values (private).
        self._last_x = numpy.infty * numpy.ones(self.nlp.n)
        self._last_obj = None
        self._last_grad_obj = None
        self._last_Hx = None


    def obj(self, x, **kwargs):
        """
        Evaluate quadratic approximation of the Augmented Lagrangian for the
        trust-region subproblem objective function at x:

        < g , x > + .5 * x' * H * x

        where `g` is the gradient of the Augmented Lagrangian evaluated at x_k
        and `H` is  the Hessian of the Augmented Lagrangian evaluated at x_k.

        """
        if self._last_obj is not None and (self._last_x == x).all():
            return self._last_obj

        if self._last_Hx is not None and (self._last_x == x).all():
            Hx = self._last_Hx.copy()
        else:
            Hx = self.nlp.hprod(self.x_k, None, x)
            self._last_Hx = Hx.copy()

        qapprox = numpy.dot(self.g_k.copy(), x)
        qapprox += .5 * numpy.dot(x, Hx)

        if not (self._last_x == x).all():
            self._last_x = x.copy()
            self._last_grad_obj = None  # Gradient out of date

        self._last_obj = qapprox
        return qapprox

    def grad(self, x, **kwargs):
        """
        """
        if self._last_grad_obj is not None and (self._last_x == x).all():
            return self._last_grad_obj

        g = self.g_k.copy()
        g += self.nlp.hprod(self.x_k, self.nlp.pi0, x)

        if not (self._last_x == x).all():
            self._last_x = x.copy()
            self._last_obj = None  # Objective function out of date
            self._last_Hx = None   # Hessian out of date

        self._last_grad_obj = g
        return g

    def hprod(self, x, pi, p, **kwargs):
        """
        """
        if self._last_Hx is not None and (self._last_x == x).all():
            return self._last_Hx

        if not (self._last_x == x).all():
            self._last_x = x.copy()
            self._last_obj = None  # Objective function out of date
            self._last_grad_obj = None  # Gradient out of date

        return self.nlp.hprod(self.x_k, pi, p)
