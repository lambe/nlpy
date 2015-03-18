"""
A pure Python/numpy implementation of the Steihaug-Toint
truncated preconditioned conjugate gradient algorithm as described in

  T. Steihaug, *The conjugate gradient method and trust regions in large scale
  optimization*, SIAM Journal on Numerical Analysis **20** (3), pp. 626-637,
  1983.

.. moduleauthor:: D. Orban <dominique.orban@gerad.ca>
"""

from nlpy.tools.exceptions import UserExitRequest
from nlpy.tools.utils import NullHandler
import numpy as np
import logging
from math import sqrt
import sys

__docformat__ = 'restructuredtext'

class TruncatedCG:

    def __init__(self, g, H, **kwargs):
        """
        Solve the quadratic trust-region subproblem

          minimize    < g, s > + 1/2 < s, Hs >
          subject to  < s, s >  <=  radius

        by means of the truncated conjugate gradient algorithm (aka the
        Steihaug-Toint algorithm). The notation `< x, y >` denotes the dot
        product of vectors `x` and `y`. `H` must be a symmetric matrix of
        appropriate size, but not necessarily positive definite.

        :returns:

          :step:       final step,
          :niter:      number of iterations,
          :stepNorm:   Euclidian norm of the step,
          :dir:        direction of infinite descent (if radius=None and
                       H is not positive definite),
          :onBoundary: set to True if trust-region boundary was hit,
          :infDescent: set to True if a direction of infinite descent was found

        The algorithm stops as soon as the preconditioned norm of the gradient
        falls under

            max( abstol, reltol * g0 )

        where g0 is the preconditioned norm of the initial gradient (or the
        Euclidian norm if no preconditioner is given), or as soon as the
        iterates cross the boundary of the trust region.
        """

        self.H = H
        self.g = g
        self.n = len(g)

        self.prefix = 'Pcg: '
        self.name = 'Truncated CG'

        self.status = '?'
        self.onBoundary = False
        self.step = None
        self.stepNorm = 0.0
        self.niter = 0
        self.dir = None

        # Formats for display
        self.hd_fmt = ' %-5s  %9s  %8s'
        self.header = self.hd_fmt % ('Iter', 'Residual', 'Curvature')
        self.fmt = ' %-5d  %9.2e  %9.2e'

        # Create a logger for solver.
        self.log = logging.getLogger('nlpy.pcg')
        try:
            self.log.addHandler(logging.NullHandler()) # For Python 2.7.x
        except:
            self.log.addHandler(NullHandler()) # For Python 2.6.x (and older?)

        return


    def to_boundary(self, s, p, radius, ss=None):
        """
        Given vectors `s` and `p` and a trust-region radius `radius` > 0,
        return the positive scalar `sigma` such that

          `|| s + sigma * p || = radius`

        in Euclidian norm. If known, supply optional argument `ss` whose value
        should be the squared Euclidian norm of `s`.
        """
        if radius is None:
            raise ValueError, 'Input value radius must be positive number.'
        sp = np.dot(s,p)
        pp = np.dot(p,p)
        if ss is None: ss = np.dot(s,s)
        sigma = (-sp + sqrt(sp*sp + pp * (radius*radius - ss)))
        sigma /= pp
        return sigma

    def post_iteration(self, *args, **kwargs):
        """
        Subclass and override this method to implement custom post-iteration
        actions. This method will be called at the end of each CG iteration.
        """
        pass

    def Solve(self, **kwargs):
        """
        Solve the trust-region subproblem.

        :keywords:

          :s0:         initial guess (default: [0,0,...,0]),
          :radius:     the trust-region radius (default: None),
          :abstol:     absolute stopping tolerance (default: 1.0e-8),
          :reltol:     relative stopping tolerance (default: 1.0e-6),
          :maxiter:    maximum number of iterations (default: 2n),
          :prec:       a user-defined preconditioner.
        """

        radius  = kwargs.get('radius', None)
        abstol  = kwargs.get('abstol', 1.0e-8)
        reltol  = kwargs.get('reltol', 1.0e-6)
        maxiter = kwargs.get('maxiter', 50*self.n)
        prec    = kwargs.get('prec', lambda v: v)
        # debug   = kwargs.get('debug', False)

        n = self.n
        g = self.g
        H = self.H

        # Initialization
        r = g.copy()
        if 's0' in kwargs:
            s = kwargs['s0']
            snorm2 = np.linalg.norm(s)
            Hs = H * s
            r += Hs                 # r = g + H s0
            Hs *= 0.5
            Hs += g
            self.qval = np.dot(s, Hs)
        else:
            s = np.zeros(n)
            snorm2 = 0.0
            self.qval = 0.0
        y = prec(r)
        ry = np.dot(r, y)

        try:
            sqrtry = sqrt(ry)
        except:
            msg = 'Preconditioned residual = %8.1e\n' % ry
            msg += 'Is preconditioner positive definite?'
            raise ValueError, msg

        stopTol = max(abstol, reltol * sqrtry)

        exitOptimal = sqrtry <= stopTol
        exitIter = exitUser = False

        # Initialize r as a copy of g not to alter the original g
        if 'p0' in kwargs:
            p = kwargs['p0']
        else:
            p = -y                       # p = - preconditioned residual
        k = 0

        onBoundary = False
        infDescent = False

        # if debug:
        self.log.info('-' * len(self.header))
        self.log.info(self.header)
        self.log.info('-' * len(self.header))
        #while sqrtry > stopTol and k < maxiter and \
        while not (exitOptimal or exitIter or exitUser) and \
                not onBoundary and not infDescent:

            k += 1
            Hp  = H * p
            pHp = np.dot(p, Hp)

            # if debug:
            self.log.info(self.fmt % (k, sqrtry, pHp))

            # Compute steplength to the boundary.
            if radius is not None:
                sigma = self.to_boundary(s, p, radius, ss=snorm2)

            # Compute CG steplength.
            if pHp == 0.:
                alpha = np.inf
            else:
                alpha = ry/pHp

            if pHp <= 0 and radius is None:
                # p is direction of singularity or negative curvature.
                self.status = 'infinite descent'
                snorm2 = 0
                self.dir = p
                self.pHp = pHp
                infDescent = True
                continue

            if radius is not None and (pHp <= 0 or alpha > sigma):
                # p leads past the trust-region boundary. Move to the boundary.
                s += sigma * p
                snorm2 = radius*radius
                #self.status = 'on boundary (sigma = %g)' % sigma
                self.status = 'trust-region boundary active'
                onBoundary = True
                continue

            # Update objective function value.
            self.qval += alpha * np.dot(r, p) + 0.5 * alpha * alpha * pHp

            # Move to next iterate.
            s += alpha * p
            r += alpha * Hp
            y = prec(r)
            ry_next = np.dot(r, y)

            if ry == 0.0 and ry_next == 0.0:
                beta = 0.0
            elif ry == 0.0:
                beta = np.inf
            else:
                beta = ry_next/ry

            #p = -y + beta * p
            p *= beta
            p -= y

            ry = ry_next

            try:
                sqrtry = sqrt(ry)
            except:
                msg = 'Preconditioned residual = %8.1e\n' % ry
                msg += 'Is preconditioner positive definite?'
                raise ValueError, msg

            snorm2 = np.dot(s,s)

            # Transfer useful quantities for post iteration.
            self.pHp = pHp
            self.p = p
            self.r = r
            self.y = y
            self.step = s
            self.stepNorm2 = snorm2
            self.beta = beta
            self.ry = ry
            self.alpha = alpha
            #self.qval = model_value(H,g,s)
            self.iter = k

            try:
                self.post_iteration()
            except UserExitRequest:
                self.status = 'usr'

            exitUser    = self.status == 'usr'
            exitIter    = k >= maxiter
            exitOptimal = sqrtry <= stopTol

        # Output info about the last iteration.
        # if debug:
        #self.log.info(self.fmt % (k, ry, pHp))
        #self.log.debug('qval: %6.2e' % self.qval)
        if k < maxiter and not onBoundary and not infDescent and not exitUser:
            self.status = 'residual small'
        elif k >= maxiter:
            self.status = 'max iter'
        self.step = s
        self.niter = k
        self.stepNorm = sqrt(snorm2)
        self.onBoundary = onBoundary
        self.infDescent = infDescent
        return


def model_value(H, g, s):
    # Return <g,s> + 1/2 <s,Hs>
    return np.dot(g,s) + 0.5 * np.dot(s, H*s)

def model_grad(H, g, s):
    # Return g + Hs
    return g + H*s

