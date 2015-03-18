"""
This module implements a matrix-free active-set method for the
trust-region quadratic program with bound constraints

    minimize  g'x + 1/2 x'Hx  subject to l <= x <= u and ||x||_2 <= Delta

where l and u define a (possibly unbounded) box. The method
combines the Steihaug-Toint CG algorithm with the projected line 
search strategy of

    J. J. More and G. Toraldo, On the solution of large
    quadratic programming problems with bound constraints,
    SIAM Journal on Optimization, 1(1), pp. 93-113, 1991.

to handle bound constraints efficiently within a Euclidean-norm 
trust region.
"""

from nlpy.krylov.pcg   import TruncatedCG
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.krylov.linop import SymmetricallyReducedLinearOperator as ReducedHessian
from nlpy.tools.utils import identical, where, NullHandler
from nlpy.tools.exceptions import InfeasibleError, UserExitRequest
import numpy as np
import logging
import warnings

class BSTCG(object):
    """
    A matrix-free trust-region solver for bound-constrained problems.
    """
    def __init__(self, qp, **kwargs):

        if qp.m != 0:
            warnings.warn(('\nYou\'re trying to solve a constrained problem '
                           'with an unconstrained solver !\n'))

        self.qp = qp
        self.Lvar = qp.Lvar
        self.Uvar = qp.Uvar
        self.H = SimpleLinearOperator(qp.n, qp.n,
                                      lambda u: self.qp.hprod(self.qp.x0,
                                                              None,
                                                              u),
                                      symmetric=True)

        self.armijo_factor = 1.0e-2
        self.cgiter = 0   # Total number of CG iterations.

        self.verbose = kwargs.get('verbose', True)
        self.hformat = '          %-5s  %9s  %8s  %5s'
        self.header  = self.hformat % ('Iter', 'q(x)', '|pg(x)|', 'cg')
        self.hlen    = len(self.header)
        self.hline   = '          ' + '-' * self.hlen
        self.format  = '          %-5d  %9.2e  %8.2e  %5d'
        self.format0 = '          %-5d  %9.2e  %8.2e  %5s'

        # Create a logger for solver.
        self.log = logging.getLogger('nlpy.bstcg')
        try:
            self.log.addHandler(logging.NullHandler()) # For Python 2.7.x
        except:
            self.log.addHandler(NullHandler()) # For Python 2.6.x (and older?)


    def check_feasible(self, x):
        """
        Safety function. Check that x is feasible with respect to the
        bound constraints.
        """
        if np.any((x < self.Lvar) | (x > self.Uvar)):
            raise InfeasibleError
        return None


    def pgrad(self, x, g=None, active_set=None, check_feasible=True):
        """
        Compute the projected gradient of the quadratic at x.
        If the actual gradient is known, it should be passed using the
        `g` keyword.
        If the active set at x0 is known, it should be passed using the
        `active_set` keyword.
        Optionally, check that x is feasible.

        The projected gradient pg is defined componentwise as follows:

        pg[i] = min(g[i],0)  if x[i] is at its lower bound,
        pg[i] = max(g[i],0)  if x[i] is at its upper bound,
        pg[i] = g[i]         otherwise.
        """
        if check_feasible: self.check_feasible(x)

        if g is None: g = self.qp.grad(x)

        if active_set is None:
            active_set = self.get_active_set(x)
        lower, upper = active_set

        pg = g.copy()
        pg[lower] = np.minimum(g[lower], 0)
        pg[upper] = np.maximum(g[upper], 0)
        return pg


    def project(self, x):
        "Project a given x into the bounds in Euclidian norm."
        return np.minimum(self.qp.Uvar,
                          np.maximum(self.qp.Lvar, x))


    def get_active_set(self, x, check_feasible=True):
        """
        Return the set of active constraints at x.
        Optionally, check that x is feasible.

        Returns the couple (lower,upper) containing the indices of variables
        that are at their lower and upper bound, respectively.
        """
        lower_active = where(x == self.Lvar)
        upper_active = where(x == self.Uvar)
        return(lower_active, upper_active)


    def breakpoints(self, x, d):
        """
        Find the smallest and largest breakpoints on the half line x + t*d.
        We assume that x is feasible. Return the smallest and largest t such
        that x + t*d lies on the boundary.
        """
        pos = where((d > 0) & (x <= self.Uvar))  # Hit the upper bound.
        neg = where((d < 0) & (x >= self.Lvar))  # Hit the lower bound.
        npos = len(pos)
        nneg = len(neg)
        if npos + nneg == 0:                    # No breakpoint.
            return (np.inf, np.inf)
        bk_min = np.inf
        bk_max = 0
        if npos > 0:
            pos_steps = (self.Uvar[pos] - x[pos]) / d[pos]
            bk_min = min(bk_min, np.min(pos_steps))
            bk_max = max(bk_max, np.max(pos_steps))
        if nneg > 0:
            neg_steps = (self.Lvar[neg] - x[neg]) / d[neg]
            bk_min = min(bk_min, np.min(neg_steps))
            bk_max = max(bk_max, np.max(neg_steps))
        bk_max = max(bk_max, bk_min)
        self.log.debug('Nearest  breakpoint: %7.1e' % bk_min)
        self.log.debug('Farthest breakpoint: %7.1e' % bk_max)
        return (bk_min, bk_max)


    def projected_linesearch(self, x, g, d, qval, step=1.0, **kwargs):
        """
        Perform an Armijo-like projected linesearch in the direction d.
        Here, x is the current iterate, g is the gradient at x,
        d is the search direction, qval is q(x) and
        step is the initial steplength.
        """

        check_feasible = kwargs.get('check_feasible', True)
        if check_feasible:
            self.check_feasible(x)

        if np.linalg.norm(d) == 0.:
            self.log.debug('Zero search direction given, exiting immediately.')
            return (x, qval, step)

        if np.dot(g, d) >= 0:
            raise ValueError('Not a descent direction.')

        qp = self.qp
        factor = self.armijo_factor

        # Obtain stepsize to nearest and farthest breakpoints.
        bk_min, bk_max = self.breakpoints(x, d)

        # if bk_min <= 0.0:
        #     raise ValueError('First breakpoint is zero.')

        self.log.debug('Projected linesearch with initial q = %7.12e' % qval)

        if kwargs.get('use_bk_min', False):
            step = bk_min

        xps = self.project(x + step * d)
        q_xps = qp.obj(xps)
        slope = np.dot(g, xps - x)

        if slope >= 0:   # May happen after a projection.
            step = min(step, bk_min)
            xps = self.project(x + step * d)
            q_xps = qp.obj(xps)
            slope = np.dot(g, xps - x)

        decrease = (q_xps < qval + factor * slope)

        if not decrease:
            # Perform projected Armijo linesearch in order to reduce the step
            # until a successful step is found.
            while not decrease and step >= bk_min:
                step /= 6
                xps = self.project(x + step * d)
                q_xps = qp.obj(xps)
                self.log.debug('  Backtracking with step = %7.1e q = %7.12e' % (step, q_xps))
                slope = np.dot(g, xps - x)
                decrease = (q_xps < qval + factor * slope)
            # end while
            if step < bk_min:
                # Quadratic interpolation to find best point
                x_bk = self.project(x + bk_min * d)
                q_bk = qp.obj(x_bk)
                slope = np.dot(g, d)
                a = (q_bk - qval - slope*bk_min)/bk_min**2
                self.log.debug('Attempt interpolation, slope = %7.1e, a = %7.1e' % (slope,a))
                if a == 0:
                    step_opt = bk_min
                else:
                    step_opt = -slope/2/a
                # end if
                if a > 0 and step_opt < bk_min:
                    step = step_opt
                    xps = self.project(x + step * d)
                    q_xps = qp.obj(xps)
                else:
                    step = bk_min
                    xps = self.project(x + bk_min * d)
                    q_xps = qp.obj(xps)
                # end if
                self.log.debug('Interpolation with optimal step = %7.1e' % step)
                self.log.debug('Interpolated q = %7.12e' % q_xps)
            # end if
        # else:
        #     # The initial step yields sufficient decrease. See if we can
        #     # find a larger step with larger decrease.
        #     if step < bk_max:
        #         increase = True
        #         x_ok = xps.copy()  # Most recent iterate satisfying Armijo.
        #         q_ok = q_xps
        #         q_prev = q_xps
        #         while increase and step <= bk_max:
        #             step *= 6
        #             xps = self.project(x + step * d)
        #             q_xps = qp.obj(xps)
        #             self.log.debug('  Extrapolating with step = %7.1e q = %7.12e' % (step, q_xps))
        #             slope = np.dot(g, xps - x)
        #             increase = slope < 0 and (q_xps < qval + factor * slope) and q_xps <= q_prev
        #             if increase:
        #                 x_ok = xps.copy()
        #                 q_ok = q_xps
        #                 q_prev = q_xps
        #             # end if
        #         # end while
        #         xps = x_ok.copy()
        #         q_xps = q_ok
        #     # end if
        # #end if

        if q_xps > qval:
            # raise ValueError('Line search returning a worse function value.')
            self.log.debug('Line search returning a worse function value. Exiting linesearch.')
            self.log.debug('Difference = %7.12e' % (q_xps - qval))
            return (x, qval, 0.0) # In the rare case that the line search fails
        self.log.debug('Projected linesearch ends with q = %7.12e' % q_xps)

        return (xps, q_xps, step)


    def solve(self, **kwargs):

        # Shortcuts for convenience
        qp = self.qp
        n = qp.n
        abstol = kwargs.get('abstol', 1.0e-7)
        reltol = kwargs.get('reltol', 1.0e-5)
        max_cgiter = kwargs.get('max_cgiter', 2*n)

        # Compute initial data.
        x = self.project(qp.x0)
        lower, upper = self.get_active_set(x)
        iter = 0

        # Compute stopping tolerance.
        q_old = qp.obj(x)
        qval = q_old
        g = qp.grad(x)
        pg = self.pgrad(x, g=g, active_set=(lower, upper))
        pgNorm = np.linalg.norm(pg)

        stoptol = reltol * pgNorm + abstol

        exitOptimal = exitIter = exitTR = False

        # Print out header and initial log.
        self.log.info(self.hline)
        self.log.info(self.header)
        self.log.info(self.hline)
        self.log.info(self.format0 % (iter, 0.0, pgNorm, ''))

        # Solve problem with TruncatedCG
        # First, identify the set of free variables (zero bound multipliers)
        on_bound = np.concatenate((lower,upper))
        zero_grad = where(pg == 0.)
        fixed_vars = np.intersect1d(on_bound,zero_grad)
        free_vars = np.setdiff1d(np.arange(n, dtype=np.int), fixed_vars)

        # Construct reduced Hessian and gradient
        ZHZ = ReducedHessian(self.H, free_vars)
        # ** could we just use pg? **
        Zg  = g[free_vars]

        # Call TruncatedCG to solve the QP, ignoring inactive bounds
        cg = TruncatedCG(Zg, ZHZ)
        cg.Solve(radius=qp.delta, abstol=abstol, reltol=reltol, 
            maxiter=max_cgiter)
        iter += 1

        msg = 'CG stops (%d its, status = %s)' % (cg.niter, cg.status)
        self.log.debug(msg)

        # Expand the search direction and handle case of negative curvature 
        d = np.zeros(n)
        d[free_vars] = cg.step

        if cg.infDescent and cg.step.size != 0 and cg.dir.size != 0:
            # msg  = 'iter :%d  Negative curvature detected' % iter
            # msg += ' (%d its)' % cg.niter
            # self.log.debug(msg)
            nc_dir = np.zeros(n)
            nc_dir[free_vars] = cg.dir
            x, qval, step = self.projected_linesearch(x, g, nc_dir, qval)
        else:
            x, qval, step = self.projected_linesearch(x, g, d, qval)

        # Update the current point
        lower, upper = self.get_active_set(x)
        g = qp.grad(x)
        pg = self.pgrad(x, g=g, active_set=(lower, upper))
        pgNorm = np.linalg.norm(pg)
        self.log.info(self.format % (iter, qval, pgNorm, cg.niter))

        exitTR = cg.infDescent or cg.onBoundary
        exitIter = cg.niter >= max_cgiter
        exitOptimal = not (exitTR or exitIter)

        # Log info and store convergence data
        self.exitOptimal = exitOptimal
        self.exitIter = exitIter
        self.exitTR = exitTR
        self.niter = 1
        self.cgiter = cg.niter
        self.x = x
        self.qval = qval
        self.lower = lower
        self.upper = upper
        return

