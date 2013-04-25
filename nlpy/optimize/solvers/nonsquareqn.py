"""
nonsquareqn.py

Classes that define quasi-Newton approximations for non-square matrices. 
These classes are useful for matrix-free NLP solvers.
"""

from nlpy.tools import norms
from nlpy.tools.timing import cputime
import numpy as np
import numpy.linalg
import logging
import sys
import pdb

__docformat__ = 'restructuredtext'

class NonsquareQuasiNewton:
    """
    This is an abstract class defining a general nonsquare quasi-Newton 
    approximation. Note that in this implementation, the full matrix is 
    stored and updated (i.e. it's not a limited memory implementation)
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        """
        Arguments:
        m = Number of rows of approximation matrix
        n = Number of columns of approximation matrix
        x = the current point; used to track function changes
        vecfunc = A function that computes the required vector function 
            values at x
        jprod = A function that computes the true product of the original 
            Jacobian with a vector
        jtprod = A function that computes the true product of the original 
            transpose Jacobian with a vector
        (access to both jprod and jtprod are critical to computing the initial
            matrix and the update)

        Optional arguments:
        slack_index = Index of the first slack variable; the slack variable 
            Jacobian is known exactly, so this part should never be updated.
        """

        # Mandatory Arguments
        self.m = m
        self.n = n
        self.jprod = jprod
        self.jtprod = jtprod
        self.vecfunc = vecfunc
        self.x = x              # Keep track of the current point for matvecs

        # Indices to handle sparse parts of the full Jacobian directly
        self.slack_index = kwargs.get('slack_index',n)
        self.sparse_index = kwargs.get('sparse_index',m)
        self.m_dense = self.sparse_index
        self.n_dense = self.slack_index

        # Initial estimate of approximation
        self.A = np.zeros([self.m_dense,self.n_dense])

        # Initial function values (uninitialized at start)
        self._vecfunc = np.zeros(m)

        # Threshold on dot product s's to accept an update of the matrix.
        self.accept_threshold = 1.0e-20

        # Keep track of number of matrix-vector products.
        self.numMatVecs = 0
        self.numRMatVecs = 0

        return


    def store(self, new_x, new_s, **kwargs):
        """
        Store the update given the primal search direction new_s and the new 
        point new_x. This prototype is overwritten in subsequent types of 
        updates.
        """
        pass


    def restart(self, x):
        """
        Restart the approximation by clearing all data on past updates.
        """
        self.x = x

        self.A = np.zeros([self.m_dense,self.n_dense])
        if self.m_dense < self.n_dense:
            unitvec = np.zeros(self.m)
            for k in range(self.m_dense):
                unitvec[k-1] = 0.
                unitvec[k] = 1.
                full_prod = self.jtprod(self.x, unitvec)
                self.A[k,:] = full_prod[:self.n_dense]
        else:
            unitvec = np.zeros(self.n)
            for k in range(self.n_dense):
                unitvec[k-1] = 0.
                unitvec[k] = 1.
                full_prod = self.jprod(self.x, unitvec)
                self.A[:,k] = full_prod[:self.m_dense]

        # Alternative test 1: A zero Jacobian
        # Nothing to do

        # Alternative test 2: An identity matrix
        # for k in range(min(self.m,self.slack_index)):
        #     self.A[k,k] = 1.

        # # Account for slack variable part of Jacobian (cheap with SlackNLP class)
        # unitvec = np.zeros(self.n)
        # for k in range(self.slack_index,self.n):
        #     unitvec[k-1] = 0.
        #     unitvec[k] = 1.
        #     self.A[:,k] = self.jprod(self.x, unitvec)

        self._vecfunc = self.vecfunc(self.x)
        return


    def matvec(self, v):
        """
        Compute a matrix-vector product between the current approximation and 
        the vector v. This function uses the numpy.dot() function, which is 
        very fast for small- and medium-sized dense matrices.
        """
        self.numMatVecs += 1
        w = self.jprod(self.x, v, sparse_only=True)
        w[:self.m_dense] += np.dot(self.A,v[:self.n_dense])
        return w


    def rmatvec(self, w):
        """
        Compute a transpose matrix-vector product between the current 
        approximation and the vector w. 
        """
        self.numRMatVecs += 1
        v = self.jtprod(self.x, w, sparse_only=True)
        # A dot-product shortcut provided w stays a vector
        v[:self.n_dense] += np.dot(w[:self.m_dense],self.A)
        return v


    def get_full_mat(self):
        """
        A utility function to return the full (dense part of the) matrix.
        """
        return self.A



class Broyden(NonsquareQuasiNewton):
    """
    This class contains the Broyden approximation to a nonsymmetric 
    and possibly nonsquare matrix. Details of the approximation are given in:

    C. G. Broyden. "A Class of Methods for Solving Nonlinear Simultaneous 
    Equations." Mathematics of Computation, 19:577-593, 1965.

    S. K. Bourji and H. F. Walker. "Least-Change Secant Updates of 
    Nonsquare Matrices." SIAM Journal on Numerical Analysis, 27:1263-1294, 
    1990.
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        NonsquareQuasiNewton.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)

    def store(self, new_x, new_s):
        """
        Store the update given the primal search direction new_s and the new 
        point new_x. Under the current scheme, the cost of this operation is 
        one constraint evaluation.
        """
        slack = self.slack_index
        sparse = self.sparse_index
        s2 = numpy.dot(new_s[:slack],new_s[:slack])
        if s2 > self.accept_threshold:
            vecfunc_new = self.vecfunc(new_x)
            new_y = vecfunc_new - self._vecfunc
            # As = np.dot(self.A[:,:slack], new_s[:slack])
            sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
            As = np.dot(self.A, new_s[:slack])
            yAs = new_y[:sparse] - As - sparse_prod[:sparse]
            self.A += np.outer(yAs, new_s[:slack]) / s2
            self._vecfunc = vecfunc_new
            self.x = new_x
        return



class modBroyden(NonsquareQuasiNewton):
    """
    This class contains a modification to Broyden's original approximation in 
    which y is replaced by Js (for true Jacobian J) in the hopes of increasing 
    the accuracy of the approximation.
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        NonsquareQuasiNewton.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)

    def store(self, new_x, new_s):
        """
        Store the update given the primal search direction new_s and the new 
        point new_x. Under the current scheme, the cost of this operation is 
        one constraint evaluation.
        """
        slack = self.slack_index
        sparse = self.sparse_index

        s2 = numpy.dot(new_s[:slack],new_s[:slack])
        if s2 > self.accept_threshold:
            # vecfunc_new = self.vecfunc(new_x)
            # new_y = vecfunc_new - self._vecfunc
            # As = np.dot(self.A[:,:slack], new_s[:slack])
            sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
            As = np.dot(self.A, new_s[:slack])
            # yAs = new_y[:sparse] - As - sparse_prod[:sparse]
            full_prod = self.jprod(self.x, new_s)
            Js = full_prod[:sparse] - sparse_prod[:sparse]

            self.A += np.outer(Js - As, new_s[:slack]) / s2
            # self._vecfunc = vecfunc_new
            self.x = new_x
        return



class directBroydenA(NonsquareQuasiNewton):
    """
    This class contains a Broyden-like update in which the search direction 
    s is replaced by the error in the gradient of the infeasibility function, 
    i.e. (J_{k+1} - A_k)^T h_{k+1}
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        NonsquareQuasiNewton.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)

    def store(self, new_x, new_s):
        """
        Store the update given the primal search direction new_s and the new 
        point new_x. Under the current scheme, the cost of this operation is 
        one constraint evaluation.
        """
        self.x = new_x
        slack = self.slack_index
        sparse = self.sparse_index

        self._vecfunc = self.vecfunc(new_x)
        ATh = np.dot(self._vecfunc[:sparse],self.A)
        full_prod = self.jtprod(self.x, self._vecfunc)
        sparse_prod = self.jtprod(self.x, self._vecfunc, sparse_only=True)
        JTh = full_prod[:sparse] - sparse_prod[:sparse]

        rho = JTh - ATh
        rho2 = numpy.dot(rho,rho)
        if rho2 > self.accept_threshold:
            rho_long = np.zeros(self.n)
            rho_long[:slack] = rho
            # vecfunc_new = self.vecfunc(new_x)
            # new_y = vecfunc_new - self._vecfunc
            # As = np.dot(self.A[:,:slack], new_s[:slack])
            Ar = np.dot(self.A, rho)
            # yAs = new_y[:sparse] - As - sparse_prod[:sparse]
            full_prod = self.jprod(self.x, rho_long)
            sparse_prod = self.jprod(self.x, rho_long, sparse_only=True)
            Jr = full_prod[:sparse] - sparse_prod[:sparse]

            self.A += np.outer(Jr - Ar, rho) / rho2
            # self._vecfunc = vecfunc_new
            # self.x = new_x
        return



class adjointBroydenA(NonsquareQuasiNewton):
    """
    This class contains an adjoint Broyden approximation to a nonsymmetric 
    and possibly nonsquare matrix. Details of these approximations are given 
    in the paper

    S. Schlenkrich, A. Griewank, and A. Walther. "On the Local Convergence of 
    Adjoint Broyden Methods" Math. Prog. A, 121:221-247, 2010.

    This class implements update (A) given in the above paper.
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        NonsquareQuasiNewton.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)


    def store(self, new_x, new_s):
        """
        Store the update given the primal search direction new_s. Under the 
        current scheme, the cost of this operation is one direct and one 
        adjoint product with the original matrix.
        """
        self.x = new_x
        slack = self.slack_index
        sparse = self.sparse_index

        # As = self.matvec(new_s)
        As = np.dot(self.A,new_s[:slack])
        # As_true = self.jprod(self.x, new_s)
        full_prod = self.jprod(self.x, new_s)
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        As_true = full_prod[:sparse] - sparse_prod[:sparse]

        sigma = As_true - As
        sigma2 = numpy.dot(sigma, sigma)
        if sigma2 > self.accept_threshold:
            sigma_long = np.zeros(self.m)
            sigma_long[:sparse] = sigma
            # ATsigma = self.rmatvec(sigma)
            ATsigma = np.dot(sigma, self.A)
            # JTsigma = self.jtprod(self.x, sigma)
            full_prod = self.jtprod(self.x, sigma_long)
            sparse_prod = self.jtprod(self.x, sigma_long, sparse_only=True)
            JTsigma = full_prod[:slack] - sparse_prod[:slack]
            tau = JTsigma - ATsigma
            self.A += np.outer(sigma,tau)/sigma2
        return



class adjointBroydenB(NonsquareQuasiNewton):
    """
    This class implements update (B), from (Schlenkrich et al., 2010) - see 
    above - which satisfies the secant condition on the Jacobian, not the 
    tangent condition.
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        NonsquareQuasiNewton.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)


    def store(self, new_x, new_s):
        """
        Store the update given the primal search direction new_s. This update 
        only requires a single adjoint product with the original matrix.
        """
        self.x = new_x
        slack = self.slack_index
        sparse = self.sparse_index

        # As = self.matvec(new_s)
        As = np.dot(self.A, new_s[:slack])
        vecfunc_new = self.vecfunc(new_x)
        new_y = vecfunc_new[:sparse] - self._vecfunc[:sparse]
        self._vecfunc = vecfunc_new
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        new_y -= sparse_prod[:sparse]

        sigma = new_y - As
        sigma2 = numpy.dot(sigma, sigma)
        if sigma2 > self.accept_threshold:
            sigma_long = np.zeros(self.m)
            sigma_long[:sparse] = sigma
            # ATsigma = self.rmatvec(sigma)
            ATsigma = np.dot(sigma, self.A)
            # JTsigma = self.jtprod(self.x, sigma)
            full_prod = self.jtprod(self.x, sigma_long)
            sparse_prod = self.jtprod(self.x, sigma_long, sparse_only=True)
            JTsigma = full_prod[:slack] - sparse_prod[:slack]
            tau = JTsigma - ATsigma
            self.A += np.outer(sigma,tau)/sigma2
        return



class adjointBroydenC(NonsquareQuasiNewton):
    """
    This class implements update (C) from (Schlenkrich et al., 2010) - see 
    above - in which the descent direction of the function 0.5*c(x)'c(x) is 
    maintained at the current point.
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        NonsquareQuasiNewton.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)


    def store(self, new_x, new_s):
        """
        Store the update given the primal search direction new_s and the new 
        point new_x. Under the current scheme, the cost of this operation is 
        one adjoint product with the original matrix and one constraint 
        evaluation (possibly a repeated call).
        """
        self.x = new_x
        slack = self.slack_index
        sparse = self.sparse_index

        self._vecfunc = self.vecfunc(new_x)
        sigma = self._vecfunc[:sparse]
        sigma2 = np.dot(sigma,sigma)
        if sigma2 >= self.accept_threshold:
            mu = self.jtprod(self.x, self._vecfunc)
            sparse_prod = self.jtprod(self.x, self._vecfunc, sparse_only=True)
            mu -= sparse_prod
            # ATs = self.rmatvec(sigma)
            ATs = np.dot(sigma, self.A)
            rho = mu[:slack] - ATs
            self.A += np.outer(sigma, rho) / sigma2
        return



class mixedBroyden(NonsquareQuasiNewton):
    """
    This class contains a combination of two updates to the Jacobian. One is 
    the Broyden-like update in which the search direction s is replaced by the 
    error in the gradient of the infeasibility function, (direct Broyden A)
    i.e. (J_{k+1} - A_k)^T h_{k+1}.

    The second is the adjoint Broyden update satisfying the secant condition, 
    i.e. adjoint Broyden B. In principle, the combination of these two should 
    work better than each one separately.
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        NonsquareQuasiNewton.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)

    def store(self, new_x, new_s):
        """
        Store the update given the primal search direction new_s and the new 
        point new_x. Under the current scheme, the cost of this operation is 
        one constraint evaluation.
        """
        self.x = new_x
        slack = self.slack_index
        sparse = self.sparse_index

        # Part 1: adjoint Broyden B update
        As = np.dot(self.A, new_s[:slack])
        vecfunc_new = self.vecfunc(new_x)
        new_y = vecfunc_new[:sparse] - self._vecfunc[:sparse]
        self._vecfunc = vecfunc_new
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        new_y -= sparse_prod[:sparse]

        sigma = new_y - As
        sigma2 = numpy.dot(sigma, sigma)
        if sigma2 > self.accept_threshold:
            sigma_long = np.zeros(self.m)
            sigma_long[:sparse] = sigma
            ATsigma = np.dot(sigma, self.A)
            full_prod = self.jtprod(self.x, sigma_long)
            sparse_prod = self.jtprod(self.x, sigma_long, sparse_only=True)
            JTsigma = full_prod[:slack] - sparse_prod[:slack]
            tau = JTsigma - ATsigma
            self.A += np.outer(sigma,tau)/sigma2


        # Part 2: direct Broyden A update
        ATh = np.dot(self._vecfunc[:sparse],self.A)
        full_prod = self.jtprod(self.x, self._vecfunc)
        sparse_prod = self.jtprod(self.x, self._vecfunc, sparse_only=True)
        JTh = full_prod[:sparse] - sparse_prod[:sparse]

        rho = JTh - ATh
        rho2 = numpy.dot(rho,rho)
        if rho2 > self.accept_threshold:
            rho_long = np.zeros(self.n)
            rho_long[:slack] = rho
            Ar = np.dot(self.A, rho)
            full_prod = self.jprod(self.x, rho_long)
            sparse_prod = self.jprod(self.x, rho_long, sparse_only=True)
            Jr = full_prod[:sparse] - sparse_prod[:sparse]
            pi = Jr - Ar
            self.A += np.outer(pi,rho)/rho2
        return



class TR1A(NonsquareQuasiNewton):
    """
    This class implements the TR1 update given in (Schlenkrich et al., 2010) 
    using the same choice of sigma as in adjointBroydenC above. This allows 
    the method to preserve the property of maintaining the correct descent 
    direction for the minimum infeasibility problem.
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        NonsquareQuasiNewton.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)
        self.accept_threshold = 1.0e-8  # Different definition for SR1 type approximations


    def store(self, new_x, new_s):
        """
        Store the update given the primal search direction and new point. 
        Under the current scheme, the cost of this operation is one direct 
        and one adjoint product with the original matrix.
        """
        slack = self.slack_index
        sparse = self.sparse_index
        self.x = new_x

        # As = self.matvec(new_s)
        As = np.dot(self.A,new_s[:slack])
        # Js = self.jprod(new_x, new_s)
        full_prod = self.jprod(self.x, new_s)
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        Js = full_prod[:sparse] - sparse_prod[:sparse]
        self._vecfunc = self.vecfunc(new_x)

        sigma = self._vecfunc[:sparse]
        denom = np.dot(sigma, Js - As)
        norm_prod = (np.dot(sigma,sigma)**0.5)*(np.dot(Js - As, Js - As)**0.5)
        if abs(denom) > self.accept_threshold*norm_prod:
            sigma_long = np.zeros(self.m)
            sigma_long[:sparse] = sigma
            # ATsigma = self.rmatvec(sigma)
            ATsigma = np.dot(sigma, self.A)
            # JTsigma = self.jtprod(new_x,sigma)
            full_prod = self.jtprod(self.x, sigma_long)
            sparse_prod = self.jtprod(self.x, sigma_long, sparse_only=True)
            JTsigma = full_prod[:slack] - sparse_prod[:slack]
            self.A += np.outer(Js - As, JTsigma - ATsigma) / denom
        return



class TR1B(TR1A):
    """
    This class implements the TR1 update given in (Schlenkrich et al., 2010) 
    using the same choice of sigma as in adjointBroydenC above. This allows 
    the method to preserve the property of maintaining the correct descent 
    direction for the minimum infeasibility problem.

    Unlike the TR1A class, TR1B enforces the adjoint secant condition, rather 
    than the adjoint tangent condition.
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        TR1A.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)


    def store(self, new_x, new_s):
        """
        Store the update given the primal search direction and new point. 
        Under the current scheme, the cost of this operation is one direct 
        and one adjoint product with the original matrix.
        """
        slack = self.slack_index
        sparse = self.sparse_index
        self.x = new_x

        # As = self.matvec(new_s)
        As = np.dot(self.A, new_s[:slack])
        vecfunc_new = self.vecfunc(new_x)
        y = vecfunc_new[:sparse] - self._vecfunc[:sparse]
        self._vecfunc = vecfunc_new
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        yAs = y - As - sparse_prod[:sparse]

        sigma = self._vecfunc[:sparse]
        denom = np.dot(sigma, yAs)
        norm_prod = (np.dot(sigma,sigma)**0.5)*(np.dot(yAs, yAs)**0.5)
        if abs(denom) > self.accept_threshold*norm_prod:
            sigma_long = np.zeros(self.m)
            sigma_long[:sparse] = sigma
            # ATsigma = self.rmatvec(sigma)
            ATsigma = np.dot(sigma, self.A)
            # JTsigma = self.jtprod(new_x,sigma)
            full_prod = self.jtprod(self.x, sigma_long)
            sparse_prod = self.jtprod(self.x, sigma_long, sparse_only=True)
            JTsigma = full_prod[:slack] - sparse_prod[:slack]
            self.A += np.outer(yAs, JTsigma - ATsigma) / denom
        return



class TR1C(TR1A):
    """
    This class implements the TR1 update given in (Schlenkrich et al., 2010) 
    using the same choice of sigma as in adjointBroydenB above. Therefore, the 
    directional derivative of the constraints will be correct along both the 
    step direction and the secant residual direction.
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        TR1A.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)


    def store(self, new_x, new_s):
        """
        Store the update given the primal search direction and new point. 
        Under the current scheme, the cost of this operation is one direct 
        and one adjoint product with the original matrix plus a function 
        evaluation.
        """
        slack = self.slack_index
        sparse = self.sparse_index
        self.x = new_x

        # As = self.matvec(new_s)
        As = np.dot(self.A,new_s[:slack])
        vecfunc_new = self.vecfunc(new_x)
        y = vecfunc_new[:sparse] - self._vecfunc[:sparse]
        self._vecfunc = vecfunc_new
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        sigma = y - As - sparse_prod[:sparse]

        # Js = self.jprod(new_x, new_s)
        full_prod = self.jprod(self.x, new_s)
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        Js = full_prod[:sparse] - sparse_prod[:sparse]

        denom = np.dot(sigma, Js - As)
        norm_prod = (np.dot(sigma,sigma)**0.5)*(np.dot(Js - As, Js - As)**0.5)
        if abs(denom) > self.accept_threshold*norm_prod:
            sigma_long = np.zeros(self.m)
            sigma_long[:sparse] = sigma
            # ATsigma = self.rmatvec(sigma)
            ATsigma = np.dot(sigma, self.A)
            # JTsigma = self.jtprod(new_x,sigma)
            full_prod = self.jtprod(self.x, sigma_long)
            sparse_prod = self.jtprod(self.x, sigma_long, sparse_only=True)
            JTsigma = full_prod[:slack] - sparse_prod[:slack]
            self.A += np.outer(Js - As, JTsigma - ATsigma) / denom
        return
            
