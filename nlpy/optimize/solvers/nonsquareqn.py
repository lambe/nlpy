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

from mpi4py import MPI

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
        # self.A = np.zeros([self.m_dense,self.n_dense])

        # Initial function values (uninitialized at start)
        self._vecfunc = np.zeros(m)

        # Threshold on dot product s's to accept an update of the matrix.
        self.accept_threshold = 1.0e-20

        # Keep track of number of matrix-vector products.
        self.numMatVecs = 0
        self.numRMatVecs = 0

        # If a warmstart is used, pull existing data from the specified file
        self.warmstart_init = kwargs.get('warmstart',False)
        self.save_data = kwargs.get('save_data',True)
        self.shelf_handle = kwargs.get('shelf_handle',None)

        # MPI data for faster matvecs and rmatvecs
        # Rougly equal partition of all rows
        self.comm = MPI.COMM_WORLD
        size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        mpi_lo = self.rank*self.m_dense/size
        mpi_hi = (self.rank+1)*self.m_dense/size
        mpi_num_rows = mpi_hi - mpi_lo

        # The part of the approximate Jacobian stored locally
        self.A_part = np.zeros([mpi_num_rows, self.n_dense])

        # Numpy index arrays for MPI functions
        self.inds = self.comm.allgather(mpi_lo)
        self.sizes = self.comm.allgather(mpi_num_rows)
        self.inds = np.array(self.inds)
        self.sizes = np.array(self.sizes)
        self.inds_full = self.inds*self.n_dense
        self.sizes_full = self.sizes*self.n_dense

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
        self._vecfunc = self.vecfunc(self.x)

        if self.warmstart_init and self.shelf_handle != None:
            # Root processor pulls parts of the Jacobian off the shelf 
            # and distributes them with point-to-point comm routines
            if self.rank == 0:
                self.A_part = self.shelf_handle['J_approx_0']
                for i in xrange(1,self.comm.Get_size()):
                    A_block = self.shelf_handle['J_approx_%d'%(i)]
                    self.comm.Send(A_block, dest=i, tag=i)
            else:
                self.A_part = np.empty([self.sizes[self.rank], self.n_dense])
                self.comm.Recv(self.A_part, source=0, tag=self.rank)

            self.warmstart_init = False  # In case another restart is needed later
        else:
            # self.A = np.zeros([self.m_dense,self.n_dense])
            self.A_part = np.zeros([self.sizes[self.rank], self.n_dense])
            if self.m_dense < self.n_dense:
                unitvec = np.zeros(self.m)
                lo_ind = self.inds[self.rank]
                hi_ind = self.inds[self.rank] + self.sizes[self.rank]
                # This code assumes that MPI is used to form the matvecs
                for k in range(self.m_dense):
                    unitvec[k-1] = 0.
                    unitvec[k] = 1.
                    full_prod = self.jtprod(self.x, unitvec)
                    if k >= lo_ind and k < hi_ind:
                        # Put the product vector in the right place
                        self.A_part[k-lo_ind,:] = full_prod[:self.n_dense]                    
            else:
                unitvec = np.zeros(self.n)
                lo_ind = self.inds[self.rank]
                hi_ind = self.inds[self.rank] + self.sizes[self.rank]
                for k in range(self.n_dense):
                    unitvec[k-1] = 0.
                    unitvec[k] = 1.
                    full_prod = self.jprod(self.x, unitvec)
                    # self.A[:,k] = full_prod[:self.m_dense]
                    self.A_part[:,k] = full_prod[lo_ind:hi_ind]

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
        # end if

        return


    def dense_matvec(self, v_block):
        """
        Compute the appropriate product with the dense block of the 
        approximation. MPI may or may not be used here.
        """
        # w = np.dot(self.A,v_block)

        # Distributed matvec
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]
        # w_block = np.dot(self.A[lo_ind:hi_ind,:],v_block)
        w_block = np.dot(self.A_part, v_block)
        w = np.zeros(self.m_dense)
        self.comm.Allgatherv([w_block, MPI.DOUBLE], [w, self.sizes, self.inds, MPI.DOUBLE])
        return w


    def dense_rmatvec(self, w_block):
        """
        Compute the appropriate product with the dense block of the 
        approximation. MPI may or may not be used here.
        """
        # v = np.dot(w_block,self.A)

        # Distributed rmatvec
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]
        # v_block = np.dot(w_block[lo_ind:hi_ind],self.A[lo_ind:hi_ind,:])
        v_block = np.dot(w_block[lo_ind:hi_ind],self.A_part)
        v = np.zeros(self.n_dense)
        self.comm.Allreduce([v_block, MPI.DOUBLE], [v, MPI.DOUBLE], MPI.SUM)
        return v


    def matvec(self, v):
        """
        Compute a matrix-vector product between the current approximation and 
        the vector v. This function uses the numpy.dot() function, which is 
        very fast for small- and medium-sized dense matrices.
        """
        self.numMatVecs += 1
        w = self.jprod(self.x, v, sparse_only=True)
        # w[:self.m_dense] += np.dot(self.A,v[:self.n_dense])
        w[:self.m_dense] += self.dense_matvec(v[:self.n_dense])
        return w


    def rmatvec(self, w):
        """
        Compute a transpose matrix-vector product between the current 
        approximation and the vector w. 
        """
        self.numRMatVecs += 1
        v = self.jtprod(self.x, w, sparse_only=True)
        # A dot-product shortcut provided w stays a vector
        # v[:self.n_dense] += np.dot(w[:self.m_dense],self.A)
        v[:self.n_dense] += self.dense_rmatvec(w[:self.m_dense])
        return v


    def get_full_mat(self):
        """
        A utility function to return the full (dense part of the) matrix.
        """
        # return self.A
        return self.A_part


    def save_mat(self):
        """
        Save the matrix to a text file in case of premature stop.
        """
        if self.save_data and self.shelf_handle != None:
            # Exectute the reverse of the retrieval operation in self.restart()
            nprocs = self.comm.Get_size()
            if self.rank == 0:
                self.shelf_handle['J_approx_0'] = self.A_part
                for i in xrange(1,nprocs):
                    A_block = np.empty([self.sizes[i], self.n_dense])
                    self.comm.Recv(A_block, source=i, tag=nprocs+i)
                    self.shelf_handle['J_approx_%d'%(i)] = A_block
                    self.shelf_handle.sync()
            else:
                self.comm.Send(self.A_part, dest=0, tag=nprocs+self.rank)
        return



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
        self.x = new_x
        slack = self.slack_index
        sparse = self.sparse_index
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]

        s2 = numpy.dot(new_s[:slack],new_s[:slack])
        if s2 > self.accept_threshold:
            vecfunc_new = self.vecfunc(new_x)
            new_y = vecfunc_new - self._vecfunc
            sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
            # As = np.dot(self.A, new_s[:slack])
            As = self.dense_matvec(new_s[:slack])
            yAs = new_y[:sparse] - As - sparse_prod[:sparse]
            # self.A += np.outer(yAs, new_s[:slack]) / s2
            self.A_part += np.outer(yAs[lo_ind:hi_ind], new_s[:slack]) / s2
            self._vecfunc = vecfunc_new
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
        self.x = new_x
        slack = self.slack_index
        sparse = self.sparse_index
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]

        s2 = numpy.dot(new_s[:slack],new_s[:slack])
        s_len = s2**0.5
        s_unit = new_s[:slack]/s_len
        if s2 > self.accept_threshold:
            s_long = np.zeros(self.n)
            s_long[:slack] = s_unit
            # As = np.dot(self.A, s_unit)
            self.dense_matvec(s_unit)
            sparse_prod = self.jprod(self.x, s_long, sparse_only=True)
            full_prod = self.jprod(self.x, s_long)
            Js = full_prod[:sparse] - sparse_prod[:sparse]
            r = Js - As
            # self.A += np.outer(r, s_unit)
            self.A_part += np.outer(r[lo_ind:hi_ind], s_unit)
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
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]

        self._vecfunc = self.vecfunc(new_x)
        # ATh = np.dot(self._vecfunc[:sparse],self.A)
        ATh = self.dense_rmatvec(self._vecfunc[:sparse])
        full_prod = self.jtprod(self.x, self._vecfunc)
        sparse_prod = self.jtprod(self.x, self._vecfunc, sparse_only=True)
        JTh = full_prod[:slack] - sparse_prod[:slack]

        rho = JTh - ATh
        rho2 = numpy.dot(rho,rho)
        rho_len = rho2**0.5
        rho_unit = rho/rho_len
        if rho2 > self.accept_threshold:
            rho_long = np.zeros(self.n)
            rho_long[:slack] = rho_unit
            # Ar = np.dot(self.A, rho_unit)
            Ar = self.dense_matvec(rho_unit)
            full_prod = self.jprod(self.x, rho_long)
            sparse_prod = self.jprod(self.x, rho_long, sparse_only=True)
            Jr = full_prod[:sparse] - sparse_prod[:sparse]
            pi = Jr - Ar
            # self.A += np.outer(pi, rho_unit)
            self.A_part += np.outer(pi[lo_ind:hi_ind], rho_unit)
        return



class adjointBroydenA(NonsquareQuasiNewton):
    """
    This class contains an adjoint Broyden approximation to a nonsymmetric 
    and possibly nonsquare matrix. Details of these approximations are given 
    in the paper

    S. Schlenkrich, A. Griewank, and A. Walther. "On the Local Convergence of 
    Adjoint Broyden Methods" Math. Prog. A, 121:221-247, 2010.

    This class implements update (A) given in the above paper. (This update is 
    structured the same as a TR1 method.)
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
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]

        # As = np.dot(self.A,new_s[:slack])
        As = self.dense_matvec(new_s[:slack])
        full_prod = self.jprod(self.x, new_s)
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        Js = full_prod[:sparse] - sparse_prod[:sparse]

        sigma = Js - As
        sigma2 = numpy.dot(sigma, sigma)
        if sigma2 > self.accept_threshold:
            sigma_len = sigma2**0.5
            sigma_unit = sigma/sigma_len
            sigma_long = np.zeros(self.m)
            sigma_long[:sparse] = sigma_unit
            # ATsigma = np.dot(sigma_unit, self.A)
            ATsigma = self.dense_rmatvec(sigma_unit)
            full_prod = self.jtprod(self.x, sigma_long)
            sparse_prod = self.jtprod(self.x, sigma_long, sparse_only=True)
            JTsigma = full_prod[:slack] - sparse_prod[:slack]
            tau = JTsigma - ATsigma
            # self.A += np.outer(sigma_unit,tau)
            self.A_part += np.outer(sigma_unit[lo_ind:hi_ind], tau)
        self.save_mat()
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
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]

        # As = np.dot(self.A, new_s[:slack])
        As = self.dense_matvec(new_s[:slack])
        vecfunc_new = self.vecfunc(new_x)
        new_y = vecfunc_new[:sparse] - self._vecfunc[:sparse]
        self._vecfunc = vecfunc_new
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        new_y -= sparse_prod[:sparse]

        sigma = new_y - As
        sigma2 = numpy.dot(sigma, sigma)
        if sigma2 > self.accept_threshold:
            sigma_len = sigma2**0.5
            sigma_unit = sigma/sigma_len
            sigma_long = np.zeros(self.m)
            sigma_long[:sparse] = sigma_unit
            # ATsigma = np.dot(sigma_unit, self.A)
            ATsigma = self.dense_rmatvec(sigma_unit)
            full_prod = self.jtprod(self.x, sigma_long)
            sparse_prod = self.jtprod(self.x, sigma_long, sparse_only=True)
            JTsigma = full_prod[:slack] - sparse_prod[:slack]
            tau = JTsigma - ATsigma
            # self.A += np.outer(sigma_unit,tau)
            self.A_part += np.outer(sigma_unit[lo_ind:hi_ind], tau)
        self.save_mat()
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

    This class provides the least-change update, satisfying both the direct 
    and adjoint tangent conditions concurrently. ** Still experimental **
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
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]

        # Part 1: form adjoint Broyden B update vectors
        # As = np.dot(self.A, new_s[:slack])
        As = self.dense_matvec(new_s[:slack])
        vecfunc_new = self.vecfunc(new_x)
        new_y = vecfunc_new[:sparse] - self._vecfunc[:sparse]
        self._vecfunc = vecfunc_new
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        new_y -= sparse_prod[:sparse]

        sigma = new_y - As
        sigma2 = numpy.dot(sigma, sigma)
        sigma_len = sigma2**0.5
        sigma_unit = sigma/sigma_len

        # Part 2: form direct Broyden A update vectors
        # ATh = np.dot(self._vecfunc[:sparse],self.A)
        ATh = self.dense_rmatvec(self._vecfunc[:sparse])
        full_prod = self.jtprod(self.x, self._vecfunc)
        sparse_prod = self.jtprod(self.x, self._vecfunc, sparse_only=True)
        JTh = full_prod[:slack] - sparse_prod[:slack]

        rho = JTh - ATh
        rho2 = numpy.dot(rho,rho)
        rho_len = rho2**0.5
        rho_unit = rho/rho_len

        # Part 3: update the approximate Jacobian
        if sigma2 > self.accept_threshold:
            sigma_long = np.zeros(self.m)
            sigma_long[:sparse] = sigma_unit
            # ATsigma = np.dot(sigma_unit, self.A)
            ATsigma = self.dense_rmatvec(sigma_unit)
            full_prod = self.jtprod(self.x, sigma_long)
            sparse_prod = self.jtprod(self.x, sigma_long, sparse_only=True)
            JTsigma = full_prod[:slack] - sparse_prod[:slack]
            tau = JTsigma - ATsigma
            # self.A += np.outer(sigma_unit,tau)
            self.A_part += np.outer(sigma_unit[lo_ind:hi_ind], tau)

        if rho2 > self.accept_threshold:
            rho_long = np.zeros(self.n)
            rho_long[:slack] = rho_unit
            # Ar = np.dot(self.A, rho_unit)
            Ar = self.dense_matvec(rho_unit)
            full_prod = self.jprod(self.x, rho_long)
            sparse_prod = self.jprod(self.x, rho_long, sparse_only=True)
            Jr = full_prod[:sparse] - sparse_prod[:sparse]
            pi = Jr - Ar
            # self.A += np.outer(pi, rho_unit)
            self.A_part += np.outer(pi[lo_ind:hi_ind], rho_unit)
            if sigma2 > self.accept_threshold:
                # self.A -= np.outer(sigma_unit,rho_unit)*np.dot(sigma_unit,pi)
                self.A_part -= np.outer(sigma_unit[lo_ind:hi_ind], rho_unit)*np.dot(sigma_unit,pi)

        return



class TR1B(NonsquareQuasiNewton):
    """
    This class implements the TR1 update given in (Schlenkrich et al., 2010) 
    using the same choice of sigma as in adjointBroydenB above. Therefore, the 
    directional derivative of the constraints will be correct along both the 
    step direction and the secant residual direction.

    Note, the corresponding "TR1A" update is identical to the adjointBroydenA 
    update presented above.
    """

    def __init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs):
        NonsquareQuasiNewton.__init__(self, m, n, x, vecfunc, jprod, jtprod, **kwargs)
        self.accept_threshold = 1.0e-8  # Different definition for SR1 type approximations


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
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]

        # As = np.dot(self.A,new_s[:slack])
        As = self.dense_matvec(new_s[:slack])
        vecfunc_new = self.vecfunc(new_x)
        y = vecfunc_new[:sparse] - self._vecfunc[:sparse]
        self._vecfunc = vecfunc_new
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        sigma = y - As - sparse_prod[:sparse]

        full_prod = self.jprod(self.x, new_s)
        sparse_prod = self.jprod(self.x, new_s, sparse_only=True)
        Js = full_prod[:sparse] - sparse_prod[:sparse]

        denom = np.dot(sigma, Js - As)
        norm_prod = (np.dot(sigma,sigma)**0.5)*(np.dot(Js - As, Js - As)**0.5)
        if abs(denom) > self.accept_threshold*norm_prod:
            sigma_long = np.zeros(self.m)
            sigma_long[:sparse] = sigma
            # ATsigma = np.dot(sigma, self.A)
            ATsigma = self.dense_rmatvec(sigma)
            full_prod = self.jtprod(self.x, sigma_long)
            sparse_prod = self.jtprod(self.x, sigma_long, sparse_only=True)
            JTsigma = full_prod[:slack] - sparse_prod[:slack]
            # self.A += np.outer(Js - As, JTsigma - ATsigma) / denom
            self.A_part += np.outer(Js[lo_ind:hi_ind] - As[lo_ind:hi_ind], JTsigma - ATsigma) / denom
        return
            


class LMadjointBroydenA(object):
    """
    This class provides a limited-memory implementation of the adjoint Broyden 
    A update. Because this class requires only a small amount of memory, no 
    MPI constructs are needed to evaluate matrix-vector products.

    ** not yet implemented **
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

        # Initial function values (uninitialized at start)
        self._vecfunc = np.zeros(m)

        # Threshold on dot product s's to accept an update of the matrix.
        self.accept_threshold = 1.0e-20

        # Keep track of number of matrix-vector products.
        self.numMatVecs = 0
        self.numRMatVecs = 0

        # Storage space for vector pairs in the limited-memory approach
        self.sigma = []
        self.sigma_bar = []
        self.q = []

        return


    def store(self, new_x, new_s, **kwargs):
        """
        Store the update given the primal search direction new_s and the new 
        point new_x. This prototype is overwritten in subsequent types of 
        updates.
        """
        # To Do
        pass


    def restart(self, x):
        """
        Restart the approximation by clear all data on past updates.
        """
        self.x = x
        self._vecfunc = self.vecfunc(self.x)

        # Loop over matrix-vector products and assemble a sparse matrix

        # Clear the set of additional stored vectors in the low-rank modification

        return


    def dense_matvec(self, v):
        """
        Compute the matrix-vector product with the dense block approximation.
        """
        w = self.A_sparse*v
        for i in xrange(self.stored_pairs):
            w += self.sigma[i]*np.dot(self.sigma_bar[i],v) - self.sigma[i]*np.dot(self.sigma[i],w)
        # end for 
        return w 


    def dense_rmatvec(self, w):
        """
        Compute the transpose matrix-vector product with the dense block 
        approximation.
        """
        v = self.A_sparse.T*w 
        for i in xrange(self.stored_pairs):
            k = np.dot(self.sigma[i],w)
            v += self.sigma_bar[i]*np.dot(self.sigma[i],w) - k*self.q[i]
        # end for 
        return v


    def matvec(self, v):
        """
        Compute a matrix-vector product between the current approximation and 
        the vector v. This function uses the numpy.dot() function, which is 
        very fast for small- and medium-sized dense matrices.
        """
        self.numMatVecs += 1
        w = self.jprod(self.x, v, sparse_only=True)
        # w[:self.m_dense] += np.dot(self.A,v[:self.n_dense])
        w[:self.m_dense] += self.dense_matvec(v[:self.n_dense])
        return w


    def rmatvec(self, w):
        """
        Compute a transpose matrix-vector product between the current 
        approximation and the vector w. 
        """
        self.numRMatVecs += 1
        v = self.jtprod(self.x, w, sparse_only=True)
        # A dot-product shortcut provided w stays a vector
        # v[:self.n_dense] += np.dot(w[:self.m_dense],self.A)
        v[:self.n_dense] += self.dense_rmatvec(w[:self.m_dense])
        return v


