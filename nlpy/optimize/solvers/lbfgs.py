from nlpy.optimize.ls.pymswolfe import StrongWolfeLineSearch
from nlpy.tools import norms
from nlpy.tools.timing import cputime
import numpy
import numpy.linalg
import logging

# import pdb

# imported packages just for testing few things:
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.optimize.solvers.lsr1 import LSR1, InverseLSR1

__docformat__ = 'restructuredtext'

class InverseLBFGS:
    """
    Class InverseLBFGS is a container used to store and manipulate
    limited-memory BFGS matrices. It may be used, e.g., in a LBFGS solver for
    unconstrained minimization or as a preconditioner. The limited-memory
    matrix that is implicitly stored is a positive definite approximation to
    the inverse Hessian. Therefore, search directions may be obtained by
    computing matrix-vector products only. Such products are efficiently
    computed by means of a two-loop recursion.

    Instantiation is as follows

    lbfgsupdate = InverseLBFGS(n)

    where n is the number of variables of the problem.

    :keywords:

        :npairs:        the number of (s,y) pairs stored (default: 5)
        :scaling:       enable scaling of the 'initial matrix'. Scaling is
                      done as 'method M3' in the LBFGS paper by Zhou and
                      Nocedal; the scaling factor is <sk,yk>/<yk,yk>
                      (default: False).

    Member functions are

    * store         to store a new (s,y) pair and discard the oldest one
                    in case the maximum storage has been reached,
    * matvec        to compute a matrix-vector product between the current
                    positive-definite approximation to the inverse Hessian
                    and a given vector.
    """

    def __init__(self, n, npairs=5, **kwargs):

        # Mandatory arguments
        self.n = n
        self.npairs = npairs

        # Optional arguments
        self.scaling = kwargs.get('scaling', False)

        # insert to points to the location where the *next* (s,y) pair
        # is to be inserted in self.s and self.y.
        self.insert = 0

        # Threshold on dot product s'y to accept a new pair (s,y).
        self.accept_threshold = 1.0e-20

        # Storage of the (s,y) pairs
        self.s = numpy.empty((self.n, self.npairs), 'd')
        self.y = numpy.empty((self.n, self.npairs), 'd')

        # Allocate two arrays once and for all:
        #  alpha contains the multipliers alpha[i]
        #  ys    contains the dot products <si,yi>
        # Only the relevant portion of each array is used
        # in the two-loop recursion.
        self.alpha = numpy.empty(self.npairs, 'd')
        self.ys = [None] * self.npairs
        self.gamma = 1.0

        # Keep track of number of matrix-vector products.
        self.numMatVecs = 0

    def store(self, new_s, new_y):
        """
        Store the new pair (new_s,new_y). A new pair
        is only accepted if the dot product <new_s, new_y> is over a certain
        threshold given by `self.accept_threshold`.
        """
        ys = numpy.dot(new_s, new_y)
        if ys > self.accept_threshold:
            insert = self.insert
            self.s[:,insert] = new_s.copy()
            self.y[:,insert] = new_y.copy()
            self.ys[insert] = ys
            self.insert += 1
            self.insert = self.insert % self.npairs
        return

    def restart(self):
        """
        Restart the approximation by clearing all data on past updates.
        """
        self.ys = [None] * self.npairs
        self.s = numpy.empty((self.n, self.npairs), 'd')
        self.y = numpy.empty((self.n, self.npairs), 'd')
        self.insert = 0
        return

    def matvec(self, v):
        """
        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the inverse Hessian matrix and the
        vector v using the LBFGS two-loop recursion formula. The 'iter'
        argument is the current iteration number.

        When the inner product <y,s> of one of the pairs is nearly zero, the
        function returns the input vector v, i.e., no preconditioning occurs.
        In this case, a safeguarding step should probably be taken.
        """
        self.numMatVecs += 1

        q = v.copy()
        s = self.s ; y = self.y ; ys = self.ys ; alpha = self.alpha
        for i in xrange(self.npairs):
            k = (self.insert - 1 - i) % self.npairs
            if ys[k] is not None:
                alpha[k] = numpy.dot(s[:,k], q)/ys[k]
                q -= alpha[k] * y[:,k]

        r = q
        if self.scaling:
            last = (self.insert - 1) % self.npairs
            if ys[last] is not None:
                self.gamma = ys[last]/numpy.dot(y[:,last],y[:,last])
                r *= self.gamma

        for i in xrange(self.npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                beta = numpy.dot(y[:,k], r)/ys[k]
                r += (alpha[k] - beta) * s[:,k]
        return r

    def solve(self, v):
        """
        This is an alias for matvec used for preconditioning.
        """
        return self.matvec(v)

    def __call__(self, v):
        """
        This is an alias for matvec.
        """
        return self.matvec(v)

    def __mult__(self, v):
        """
        This is an alias for matvec.
        """
        return self.matvec(v)


class LBFGS(InverseLBFGS):
    """
    Class LBFGS is similar to InverseLBFGS, except that it operates 
    on the Hessian approximation directly, rather than forming the inverse. 
    Additional information is stored to compute this approximation 
    efficiently.

    This form is useful in trust region methods, where the approximate Hessian 
    is used in the model problem.
    """

    def __init__(self, n, npairs=5, **kwargs):
        InverseLBFGS.__init__(self, n, npairs, **kwargs)

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlpy.lbfgs')
        self.log = logging.getLogger(logger_name)
        #self.log.addHandler(logging.NullHandler())
        self.log.info('Logger created')

    def matvec(self, v):
        """
        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the direct Hessian matrix and the
        vector v using the outer product representation.

        Note: there is probably some optimization that could be done in this 
        function with respect to memory use and storing key dot products.
        """
        self.numMatVecs += 1

        q = v.copy()
        r = v.copy()
        s = self.s ; y = self.y ; ys = self.ys
        prodn = 2*self.npairs
        a = numpy.zeros(prodn,'d')
        minimat = numpy.zeros([prodn,prodn],'d')

        if self.scaling:
            last = (self.insert - 1) % self.npairs
            if ys[last] is not None:
                self.gamma = ys[last]/numpy.dot(y[:,last],y[:,last])
                r /= self.gamma

        paircount = 0
        for i in xrange(self.npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                a[paircount] = numpy.dot(r[:],s[:,k])
                paircount += 1

        j = 0
        for i in xrange(self.npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                a[paircount+j] = numpy.dot(q[:],y[:,k])
                j += 1

        # Populate small matrix to be inverted
        k_ind = 0
        for i in xrange(self.npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                minimat[paircount+k_ind,paircount+k_ind] = -ys[k]
                minimat[k_ind,k_ind] = numpy.dot(s[:,k],s[:,k])/self.gamma
                l_ind = 0
                for j in xrange(i):
                    l = (self.insert + j) % self.npairs
                    if ys[l] is not None:
                        minimat[k_ind,paircount+l_ind] = numpy.dot(s[:,k],y[:,l])
                        minimat[paircount+l_ind,k_ind] = minimat[k_ind,paircount+l_ind]
                        minimat[k_ind,l_ind] = numpy.dot(s[:,k],s[:,l])/self.gamma
                        minimat[l_ind,k_ind] = minimat[k_ind,l_ind]
                        l_ind += 1
                k_ind += 1

        if paircount > 0:
            rng = 2*paircount
            b = numpy.linalg.solve(minimat[0:rng,0:rng],a[0:rng])

        for i in xrange(paircount):
            k = (self.insert - paircount + i) % self.npairs
            r -= (b[i]/self.gamma)*s[:,k]
            r -= b[i+paircount]*y[:,k]

        return r

    def store(self, new_s, new_y):
        InverseLBFGS.store(self,new_s,new_y)
        ys = numpy.dot(new_s, new_y)
        if ys < self.accept_threshold:
            self.log.debug('Not accepting LBFGS update: ys = %g' % ys)
        return

class LBFGS_unrolling(InverseLBFGS):
    """
    Class LBFGS is similar to InverseLBFGS, except that it operates 
    on the Hessian approximation directly, rather than forming the inverse.
    Additional information is stored to compute this approximation 
    efficiently.

    This form is useful in trust region methods, where the approximate Hessian
    is used in the model problem.
    """

    def __init__(self, n, npairs=5, **kwargs):
        InverseLBFGS.__init__(self, n, npairs, **kwargs)

        # Setup the logger. Install a NullHandler if no output needed
        logger_name = kwargs.get('logger_name', 'nlpy.lbfgs')
        self.log = logging.getLogger(logger_name)
        #self.log.addHandler(logging.NullHandler())
        self.log.info('Logger created')

    def matvec(self, v):
        """
        Compute a matrix-vector product between the current limited-memory
        positive-definite approximation to the direct Hessian matrix and the
        vector v using the outer product representation.

        Note: there is probably some optimization that could be done in this 
        function with respect to memory use and storing key dot products.
        """
        self.numMatVecs += 1

        q = v.copy()
        s = self.s ; y = self.y ; ys = self.ys
        b = numpy.zeros((self.n, self.npairs), 'd')
        a = numpy.zeros((self.n, self.npairs), 'd')

        paircount = 0
        for i in xrange(self.npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                b[:,k] = y[:,k] / ys[k]**.5
                bv = numpy.dot(b[:,k], v[:])
                q += bv * b[:,k]
                a[:,k] = s[:,k].copy()
                for j in xrange(i):
                    l = (self.insert + j) % self.npairs
                    if ys[l] is not None:
                        a[:,k] += numpy.dot(b[:,l], s[:,k]) * b[:,l]
                        a[:,k] -= numpy.dot(a[:,l], s[:,k]) * a[:,l]
                a[:,k] /= numpy.dot(s[:,k], a[:,k])**.5
                q -= numpy.dot(numpy.outer(a[:,k],a[:,k]), v[:])#numpy.dot(a[:,k], v[:]) * a[:,k]
                paircount += 1

        return q

    def store(self, new_s, new_y):
        InverseLBFGS.store(self,new_s,new_y)
        ys = numpy.dot(new_s, new_y)
        if ys < self.accept_threshold:
            self.log.debug('Not accepting LBFGS update: ys = %g' % ys)
        return

class LBFGS_structured(InverseLBFGS):
    """
    LBFGS quasi newton using an unrolling formula.
    For this procedure see [Nocedal06]
    """
    def __init__(self, n, npairs=5, **kwargs):
        InverseLBFGS.__init__(self, n, npairs, **kwargs)
        self.yd = numpy.empty((self.n, self.npairs), 'd')
        self.accept_threshold = 1e-8
        # Setup the logger. Install a NullHandler if no output needed
        logger_name = kwargs.get('logger_name', 'nlpy.lbfgs')
        self.log = logging.getLogger(logger_name)
        #self.log.addHandler(logging.NullHandler())
        self.log.info('Logger created')


    def matvec(self, v):
        """
        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        self.numMatVecs += 1

        q = v.copy()
        s = self.s ; y = self.y ; yd = self.yd ; ys = self.ys
        npairs = self.npairs
        a = numpy.zeros([self.n, npairs])
        ad = numpy.zeros([self.n, npairs])

        aTs = numpy.zeros([npairs,1])
        adTs = numpy.zeros([npairs,1])

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last]/numpy.dot(y[:,last],y[:,last])
                q /= self.gamma

        for i in xrange(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                coef = (self.gamma*ys[k]/numpy.dot(s[:,k],s[:,k]))**0.5
                a[:,k] = y[:,k] + coef * s[:,k]/self.gamma
                ad[:,k] = yd[:,k] - s[:,k]/self.gamma
                for j in xrange(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        alTs = numpy.dot(a[:,l], s[:,k])/aTs[l]
                        adlTs = numpy.dot(ad[:,l], s[:,k])
                        update = alTs/aTs[l] * ad[:,l] + adlTs/aTs[l] * a[:,l] - adTs[l]/aTs[l] * alTs * a[:,l]
                        a[:,k] += coef * update.copy()
                        ad[:,k] -= update.copy()
                aTs[k] = numpy.dot(a[:,k], s[:,k])
                adTs[k] = numpy.dot(ad[:,k], s[:,k])
                aTv = numpy.dot(a[:,k],v[:])
                adTv = numpy.dot(ad[:,k],v[:])
                q += aTv/aTs[k] * ad[:,k] + adTv/aTs[k] * a[:,k] - aTv*adTs[k]/aTs[k]**2 * a[:,k]
        return q

    def store(self, new_s, new_y, new_yd):
        """
        Store the new pair (new_s,new_y, new_yd). A new pair
        is only accepted if 
        | y_k' s_k + (y's s_k' B_k s_k)**.5 | >= 1e-8.
        """
        ys = numpy.dot(new_s, new_y)
        Bs = self.matvec(new_s)
        ypBs = ys + (ys * numpy.dot(new_s, Bs))**0.5

        if ypBs>=self.accept_threshold:
            insert = self.insert
            self.s[:,insert] = new_s.copy()
            self.y[:,insert] = new_y.copy()
            self.yd[:,insert] = new_yd.copy()
            self.ys[insert] = ys
            self.insert += 1
            self.insert = self.insert % self.npairs
        else:
            self.log.debug('Not accepting LBFGS update')
        return



class LBFGS_new(object):
    """
    Class LBFGS_new is an experimental version that uses the unrolling formula 
    for the LBFGS approximation and a novel storage scheme to accelerate the 
    matrix-vector product computation.
    """

    def __init__(self, n, npairs=5, **kwargs):
        # Mandatory arguments
        self.n = n
        self.npairs = npairs

        # An initialization for the main diagonal, not used in the base class
        self.diag = numpy.ones(n)

        # Optional arguments
        self.scaling = kwargs.get('scaling', False)

        # The number of vector pairs that are actually stored
        self.stored_pairs = 0

        # Threshold on dot product s'y to accept a new pair (s,y).
        self.accept_threshold = 1.0e-20

        # Storage of the (s,y) pairs
        self.s = []
        self.y = []
        self.a = [None]*npairs     # Unrolled vectors for the matvec
        self.b = [None]*npairs
        self.gamma = 1.0

        # Keep track of number of matrix-vector products.
        self.numMatVecs = 0

        logger_name = kwargs.get('logger_name', 'nlpy.lbfgs')
        self.log = logging.getLogger(logger_name)
        #self.log.addHandler(logging.NullHandler())
        self.log.info('Logger created')


    def store(self, new_s, new_y):
        """
        Store the new pair (new_s,new_y). A new pair
        is only accepted if 
        s_k' y_k >= `self.accept_threshold`.
        """
        ys = numpy.dot(new_s, new_y)
        if ys > self.accept_threshold:
            self.s.append(new_s.copy())
            self.y.append(new_y.copy())
            if len(self.s) > self.npairs:
                del self.s[0]
                del self.y[0]
            else:
                self.stored_pairs += 1
            # end if

            if self.scaling:
                self.gamma = ys / numpy.dot(new_y, new_y)

            # Recompute stored data for the matvec computation
            for i in xrange(self.stored_pairs):
                self.b[i] = numpy.dot(self.y[i],self.s[i])**(-0.5) * self.y[i]
                self.a[i] = self.s[i]*self.diag/self.gamma
                for j in xrange(i):
                    bTs = numpy.dot(self.b[j],self.s[i])
                    aTs = numpy.dot(self.a[j],self.s[i])
                    self.a[i] += bTs*self.b[j] - aTs*self.a[j]
                self.a[i] *= numpy.dot(self.a[i], self.s[i])**-0.5
            # end for
        else:
            self.log.debug('Not accepting LBFGS update: ys=%g' % ys)
        return


    def restart(self):
        """
        Restart the approximation by clearing all data on past updates.
        """
        self.gamma = 1.0
        self.diag = numpy.ones(self.n)
        self.s = []
        self.y = []
        self.a = [None]*self.npairs
        self.b = [None]*self.npairs
        self.stored_pairs = 0
        return


    def matvec(self, v):
        """
        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the unrolling formula.
        """
        self.numMatVecs += 1

        w = v * self.diag / self.gamma
        for i in xrange(self.stored_pairs):
            w += numpy.dot(self.b[i],v)*self.b[i] - numpy.dot(self.a[i],v)*self.a[i]
        # end for
        return w



class LBFGS_structured_new(LBFGS_new):
    """
    A structured LBFGS approximation based on the LBFGS_new class.
    """
    def __init__(self, n, npairs=5, **kwargs):
        LBFGS_new.__init__(self, n, npairs, **kwargs)
        self.yd = []
        self.ad = self.b # A space saver, since we do not need the b list
        self.aTs = [None]*self.npairs
        self.adTs = [None]*self.npairs


    def store(self, new_s, new_y, new_yd):
        """
        Store the new pair (new_s,new_y). A new pair
        is only accepted if 
        s_k' y_k >= `self.accept_threshold`.
        """
        ys = numpy.dot(new_s, new_y)
        if ys > self.accept_threshold:
            self.s.append(new_s.copy())
            self.y.append(new_y.copy())
            self.yd.append(new_yd.copy())            
            if len(self.s) > self.npairs:
                del self.s[0]
                del self.y[0]
                del self.yd[0]
            else:
                self.stored_pairs += 1
            # end if

            if self.scaling:
                # ** This scaling criterion should probably change for the structured update
                self.gamma = ys / numpy.dot(new_y, new_y)

            # Recompute stored data for the matvec computation
            for i in xrange(self.stored_pairs):
                self.a[i] = self.s[i]/self.gamma
                self.ad[i] = self.yd[i] - self.s[i]/self.gamma
                for j in xrange(i):
                    aTs_temp = numpy.dot(self.a[j],self.s[i])
                    adTs_temp = numpy.dot(self.ad[j],self.s[i])
                    Delta_s = (aTs_temp/self.aTs[j])*self.ad[j] + (adTs_temp/self.aTs[j])*self.a[j]
                    Delta_s -= (aTs_temp*self.adTs[j]/self.aTs[j]**2)*self.a[j]
                    self.a[i] += Delta_s
                    self.ad[i] -= Delta_s
                # end for
                coeff = (numpy.dot(self.y[i],self.s[i])/numpy.dot(self.s[i],self.a[i]))**0.5
                self.a[i] = self.y[i] + coeff*self.a[i]
                self.aTs[i] = numpy.dot(self.a[i], self.s[i])
                self.adTs[i] = numpy.dot(self.ad[i], self.s[i])
            # end for
        else:
            self.log.debug('Not accepting LBFGS update: ys=%g' % ys)
        return


    def restart(self):
        """
        Restart the approximation by clearing all data on past updates.
        """
        LBFGS_new.restart(self)
        self.yd = []
        # self.ad is cleared when self.b is cleared in the parent method
        self.adTs = [None]*self.npairs
        return        


    def matvec(self, v):
        """
        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the unrolling formula.
        """
        self.numMatVecs += 1

        w = v / self.gamma
        for i in xrange(self.stored_pairs):
            aTv = numpy.dot(self.a[i],v)
            adTv = numpy.dot(self.ad[i],v)
            w += (aTv/self.aTs[i])*self.ad[i] + (adTv/self.aTs[i])*self.a[i]
            w -= (aTv*self.adTs[i]/self.aTs[i]**2)*self.a[i]
        # end for
        return w



class LBFGS_infeas(LBFGS_new):
    """
    This is a specialized LBFGS approximation for the infeasibility term of 
    an augmented Lagrangian function. The class contains specialized methods 
    for defining an initial diagonal.
    """

    def __init__(self, n, x, jprod, jtprod, npairs=5, **kwargs):
        LBFGS_new.__init__(self, n, npairs, **kwargs)
        self.slack_index = kwargs.get('slack_index',n)
        self.jprod = jprod
        self.jtprod = jtprod
        self.x = x  # Point at which to compute diagonal
        # self.compute_diag()
        self.diag_eps = 1e-6
        self.beta = kwargs.get('beta',min(self.slack_index,3))
        self.compute_diag()


    def restart(self, x):
        """
        Restart the approximation.
        """
        LBFGS_new.restart(self)
        self.x = x
        self.compute_diag()
        return


    def compute_diag(self):
        """
        Compute an estimate of the initial diagonal to seed this approximation.

        The initial diagonal comes from estimating the diagonal elements of 
        J'J, where J is the Jacobian of constraints with respect to decision 
        variables. The diagonal elements corresponding to slack variables 
        remain as an identity block.
        """
        # Safety check that product functions are defined
        if self.jprod == None or self.jtprod == None:
            return

        # Simple version - assume underlying matrix is diagonal
        # ones_vec = numpy.ones(self.n)
        # Jv = self.jprod(self.x, ones_vec)
        # JTJv = self.jtprod(self.x, Jv)
        # n_dense = self.slack_index
        # self.diag[:n_dense] = JTJv[:n_dense]
        # for i in xrange(n_dense):
        #     if self.diag[i] < self.diag_eps:
        #         self.diag[i] = max(abs(self.diag[i]),self.diag_eps)

        # More complicated schemes extract the main diagonal from a banded approximation
        for i in xrange(self.beta):
            # ind_set = numpy.arange(i,self.n,self.beta)
            range_arr = numpy.arange(self.n)
            binary_arr = numpy.where(range_arr % self.beta == i, 1, 0)
            binary_arr[self.slack_index:] = 0.
            Jv = self.jprod(self.x, binary_arr)
            JTJv = self.jtprod(self.x, Jv)
            for j in xrange(i,self.slack_index,self.beta):
                # if JTJv[j] < self.diag_eps:
                self.diag[j] = max(abs(JTJv[j]),self.diag_eps)
                # else:


        return


    # An alternative to storing infeasibility gradients
    # def store(self, new_s, new_yi, new_x):
    #     """
    #     Alternative storage scheme: pass J'Js in as the y-vector of the quasi-
    #     Newton update.
    #     """
    #     self.x = new_x
    #     Js = self.jprod(self.x, new_s)
    #     JTJs = self.jtprod(self.x, Js)
    #     LBFGS_new.store(self, new_s, JTJs)
    #     return


    # This matvec function seems to do poorly...
    # def matvec(self, v):
    #     """
    #     This matvec function is more complicated than that of the base class
    #     to preserve the identity block from the slack variables.
    #     """
    #     self.numMatVecs += 1

    #     w = v * self.diag / self.gamma
    #     for i in xrange(self.stored_pairs):
    #         # w += numpy.dot(self.b[i],v)*self.b[i] - numpy.dot(self.a[i],v)*self.a[i]
    #         w += numpy.dot(self.b[i][:self.slack_index],v[:self.slack_index])*self.b[i] - \
    #             numpy.dot(self.a[i][:self.slack_index],v[:self.slack_index])*self.a[i]
    #         w[:self.slack_index] += numpy.dot(self.b[i][self.slack_index:],v[self.slack_index:])*self.b[i][:self.slack_index] - \
    #             numpy.dot(self.a[i][self.slack_index:],v[self.slack_index:])*self.a[i][:self.slack_index]
    #     # end for
    #     return w



class LBFGS_infeas2(LBFGS_new):
    """
    This is a container class for an LBFGS method within a larger matrix.
    The idea is to approximate the infeasibility Hessian by an LBFGS matrix 
    for the second derivatives with respect to x, by an identity matrix for 
    the second derivatives with respect to the slacks, and by zero for the 
    cross-derivative terms.
    """

    def __init__(self, n, npairs=5, **kwargs):
        # pdb.set_trace()
        LBFGS_new.__init__(self, n, npairs, **kwargs)
        # Note that the update vectors must be of length n, not length big_n
        self.big_n = kwargs.get('big_n',n)


    def matvec(self,v):
        """
        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the unrolling formula.
        """
        self.numMatVecs += 1

        w = v / self.gamma
        w[self.n:] = v[self.n:]
        for i in xrange(self.stored_pairs):
            w[:self.n] += numpy.dot(self.b[i],v[:self.n])*self.b[i] - \
                numpy.dot(self.a[i],v[:self.n])*self.a[i]
        # end for
        return w



class LBFGSFramework:
    """
    Class LBFGSFramework provides a framework for solving unconstrained
    optimization problems by means of the limited-memory BFGS method.

    Instantiation is done by

    lbfgs = LBFGSFramework(nlp)

    where nlp is an instance of a nonlinear problem. A solution of the
    problem is obtained by called the solve member function, as in

    lbfgs.solve().

    :keywords:

        :npairs:    the number of (s,y) pairs to store (default: 5)
        :x0:        the starting point (default: nlp.x0)
        :maxiter:   the maximum number of iterations (default: max(10n,1000))
        :abstol:    absolute stopping tolerance (default: 1.0e-6)
        :reltol:    relative stopping tolerance (default: `nlp.stop_d`)

    Other keyword arguments will be passed to InverseLBFGS.

    The linesearch used in this version is Jorge Nocedal's modified More and
    Thuente linesearch, attempting to ensure satisfaction of the strong Wolfe
    conditions. The modifications attempt to limit the effects of rounding
    error inherent to the More and Thuente linesearch.
    """
    def __init__(self, nlp, **kwargs):

        self.nlp = nlp
        self.npairs = kwargs.get('npairs', 5)
        self.silent = kwargs.get('silent', False)
        self.abstol = kwargs.get('abstol', 1.0e-6)
        self.reltol = kwargs.get('reltol', self.nlp.stop_d)
        self.iter   = 0
        self.nresets = 0
        self.converged = False

        self.lbfgs = InverseLBFGS(self.nlp.n, **kwargs)

        self.x = kwargs.get('x0', self.nlp.x0)
        self.f = self.nlp.obj(self.x)
        self.g = self.nlp.grad(self.x)
        self.gnorm = norms.norm2(self.g)
        self.f0 = self.f
        self.g0 = self.gnorm

        # Optional arguments
        self.maxiter = kwargs.get('maxiter', max(10*self.nlp.n, 1000))
        self.tsolve = 0.0


    def solve(self):

        tstart = cputime()

        # Initial LBFGS matrix is the identity. In other words,
        # the initial search direction is the steepest descent direction.

        # This is the original L-BFGS stopping condition.
        #stoptol = self.nlp.stop_d * max(1.0, norms.norm2(self.x))
        stoptol = max(self.abstol, self.reltol * self.g0)

        while self.gnorm > stoptol and self.iter < self.maxiter:

            if not self.silent:
                print '%-5d  %-12g  %-12g' % (self.iter, self.f, self.gnorm)

            # Obtain search direction
            d = self.lbfgs.matvec(-self.g)

            # Prepare for modified More-Thuente linesearch
            if self.iter == 0:
                stp0 = 1.0/self.gnorm
            else:
                stp0 = 1.0
            SWLS = StrongWolfeLineSearch(self.f,
                                         self.x,
                                         self.g,
                                         d,
                                         lambda z: self.nlp.obj(z),
                                         lambda z: self.nlp.grad(z),
                                         stp = stp0)
            # Perform linesearch
            SWLS.search()

            # SWLS.x  contains the new iterate
            # SWLS.g  contains the objective gradient at the new iterate
            # SWLS.f  contains the objective value at the new iterate
            s = SWLS.x - self.x
            self.x = SWLS.x
            y = SWLS.g - self.g
            self.g = SWLS.g
            self.gnorm = norms.norm2(self.g)
            self.f = SWLS.f
            #stoptol = self.nlp.stop_d * max(1.0, norms.norm2(self.x))

            # Update inverse Hessian approximation using the most recent pair
            self.lbfgs.store(s, y)
            self.iter += 1


        self.tsolve = cputime() - tstart
        self.converged = (self.iter < self.maxiter)

