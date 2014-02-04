"""
fullqn_mpi.py

Implementations of full-memory quasi-Newton methods that exploit distributed
memory parallel computing using MPI instructions.
"""

from nlpy.tools import norms
from nlpy.tools.timing import cputime
import numpy as np
import numpy.linalg
import logging
import sys

from mpi4py import MPI

__docformat__ = 'restructuredtext'

class BFGS(object):
    """
    Class BFGS stores a full-memory BFGS Hessian approximation matrix. Matrix-
    vector products are computed in the usual manner, without considering 
    sparsity.

    This class supports MPI operations for fast matvecs and distributed 
    storage of the matrix.
    """

    def __init__(self, n, **kwargs):

        self.n = n  # The size of the matrix (square)
        self.accept_threshold = 1.0e-20

        # MPI data for parallel matvecs
        self.comm = MPI.COMM_WORLD
        size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        mpi_lo = self.rank*self.n/size
        mpi_hi = (self.rank+1)*self.n/size
        mpi_num_rows = mpi_hi - mpi_lo

        self.B_part = np.zeros([mpi_num_rows, self.n])  # The actual matrix

        # Numpy index arrays for MPI functions
        self.inds = self.comm.allgather(mpi_lo)
        self.sizes = self.comm.allgather(mpi_num_rows)
        self.inds = np.array(self.inds)
        self.sizes = np.array(self.sizes)
        self.inds_full = self.inds*self.n_dense
        self.sizes_full = self.sizes*self.n_dense

        # Initialize the quasi-Newton matrix as an identity matrix
        for i in xrange(mpi_num_rows):
            self.B_part[i,self.inds[self.rank]+i] = 1.0

        # A counter for the number of matrix-vector products
        self.numMatVecs = 0

        return


    def store(self, new_s, new_y):
        """
        Store the new pair (new_s,new_y). A new pair
        is only accepted if the dot product <new_s, new_y> is over a certain
        threshold given by `self.accept_threshold`.
        """
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]

        ys = numpy.dot(new_s, new_y)
        if ys > self.accept_threshold:
            # code to update self.B_part
            Bs = self.matvec(new_s)
            sBs = np.dot(new_s,Bs)
            self.B_part -= np.outer(Bs[lo_ind:hi_ind],Bs/sBs)
            self.B_part += np.outer(new_y[lo_ind:hi_ind],new_y/ys)
        # end if 

        return


    def restart(self):
        """
        Remove all past information and start with a new identity matrix.
        """

        self.B_part = np.zeros([self.sizes[self.rank], self.n])
        for i in xrange(mpi_num_rows):
            self.B_part[i,self.inds[self.rank]+i] = 1.0
        # end for 
        return


    def matvec(self, v):
        """
        Multiply the matrix by the vector v and return the result.
        """

        # Distributed matvec
        # lo_ind = self.inds[self.rank]
        # hi_ind = self.inds[self.rank] + self.sizes[self.rank]
        w_block = np.dot(self.B_part, v)
        w = np.zeros(self.n)
        self.comm.Allgatherv([w_block, MPI.DOUBLE], [w, self.sizes, self.inds, MPI.DOUBLE])

        return w 


    def rmatvec(self, w):
        """
        Since the matrix is symmetric, return the result of matvec().
        """
        return self.matvec(w)


# end class 



class SR1(BFGS):
    """
    This class is identical to the BFGS class, but implements the SR1 Hessian 
    approximation method.
    """

    def __init__(self, n, **kwargs):
        BFGS.__init__(self, n, **kwargs)
        self.accept_threshold = 1.0e-8


    def store(self, new_s, new_y):
        """
        Store the (s,y) pair only if 
        | s_k' (y_k -B_k s_k) | >= 1e-8 ||s_k|| ||y_k - B_k s_k ||.
        """
        lo_ind = self.inds[self.rank]
        hi_ind = self.inds[self.rank] + self.sizes[self.rank]

        Bs = self.matvec(new_s)
        yBs = new_y - Bs
        sTyBs = np.dot(yBs,new_s)
        criterion = abs(sTyBs) >= self.accept_threshold * np.linalg.norm(new_s) * np.linalg.norm(yBs)

        if criterion:
            self.B_part += np.outer(yBs[lo_ind,hi_ind],yBs/sTyBs)
        # end if 

        return


# end class 