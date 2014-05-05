"""
lsr1.py

A class containing a limited-memory Symmetric Rank-one (LSR1) approximation
to a symmetric matrix. This approximation may not be positive-definite and is
useful in trust region methods for optimization.
"""

# =============================================================================
# External Python modules
# =============================================================================
import numpy as np
import numpy.linalg
import logging

# =============================================================================
# LSR1 Class
# =============================================================================
class LSR1(object):
    """
    Class LSR1 is similar to LBFGS, except that it uses a different
    approximation scheme. Inheritance is currently taken through InverseLBFGS
    to avoid multiple-inheritance confusion.

    This class is useful in trust region methods, where the approximate Hessian
    is used in the model problem. LSR1 has the advantatge over LBFGS and LDFP
    of permitting approximations that are not positive-definite.
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
        self.accept_threshold = 1.0e-8

        # Storage of the (s,y) pairs
        self.s = np.zeros([self.n, self.npairs], 'd')
        self.y = np.zeros([self.n, self.npairs], 'd')
        self.alpha = np.empty(self.npairs, 'd')
        self.ys = [None] * self.npairs
        self.gamma = 1.0

        # Keep track of number of matrix-vector products.
        self.numMatVecs = 0

        logger_name = kwargs.get('logger_name', 'nlpy.lsr1')
        self.log = logging.getLogger(logger_name)
        #self.log.addHandler(logging.NullHandler())
        self.log.info('Logger created')

    def store(self, new_s, new_y):
        """
        Store the new pair (new_s,new_y). A new pair
        is only accepted if 
        | s_k' (y_k -B_k s_k) | >= 1e-8 ||s_k|| ||y_k - B_k s_k ||.
        """
        Bs = self.matvec(new_s)
        ymBs = new_y - Bs
        criterion = abs(np.dot(ymBs, new_s)) >= self.accept_threshold * np.linalg.norm(new_s) * np.linalg.norm(ymBs)
        ymBsTs_criterion = abs(np.dot(ymBs, new_s)) >= 1e-15
        ys = np.dot(new_s, new_y)

        ys_criterion = True; scaling_criterion = True; yms_criterion = True
        if self.scaling:
            if abs(ys) >= 1e-15:
                scaling_factor = ys/np.dot(new_y, new_y)
                scaling_criterion = np.linalg.norm(new_y - new_s / scaling_factor) >= 1e-10
            else:
                ys_criterion = False
        else:
            if np.linalg.norm(new_y - new_s) < 1e-10:
                yms_criterion = False

        if ymBsTs_criterion and yms_criterion and scaling_criterion and criterion and ys_criterion:
            insert = self.insert
            self.s[:,insert] = new_s.copy()
            self.y[:,insert] = new_y.copy()
            self.ys[insert] = ys
            self.insert += 1
            self.insert = self.insert % self.npairs
        else:
            self.log.debug('Not accepting LSR1 update: |<y-Bs,s>|= %s, y-s/gamma=%s, y-s = %s, ys=%s' % (criterion, scaling_criterion, yms_criterion, ys_criterion))
        return


    def restart(self):
        """
        Restart the approximation by clearing all data on past updates.
        """
        self.ys = [None] * self.npairs
        self.s = np.zeros([self.n, self.npairs], 'd')
        self.y = np.zeros([self.n, self.npairs], 'd')
        self.insert = 0
        return


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
        s = self.s ; y = self.y ; ys = self.ys
        npairs = self.npairs
        a = np.zeros([npairs,1],'d')
        minimat = np.zeros([npairs,npairs],'d')

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last]/np.dot(y[:,last],y[:,last])
                q /= self.gamma

        paircount = 0
        for i in xrange(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                a[k] = np.dot(y[:,k],v[:]) - np.dot(s[:,k],q[:])
                paircount += 1

        # Populate small matrix to be inverted
        for i in xrange(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                minimat[k,k] = ys[k] - np.dot(s[:,k],s[:,k])/self.gamma
                for j in xrange(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        minimat[k,l] = np.dot(s[:,k],y[:,l]) - np.dot(s[:,k],s[:,l])/self.gamma
                        minimat[l,k] = minimat[k,l]

        if paircount > 0:
            rng = paircount
            b = np.linalg.solve(minimat[0:rng,0:rng],a[0:rng])

        for i in xrange(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                q += b[k]*y[:,k] - b[k]/self.gamma*s[:,k]
        return q


class LSR1_unrolling(LSR1):
    """
    LSR1 quasi newton using an unrolling formula.
    For this procedure see [Nocedal06]
    """
    def __init__(self, n, npairs=5, **kwargs):
        LSR1.__init__(self, n, npairs, **kwargs)
        self.accept_threshold = 1e-8

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
        s = self.s ; y = self.y ; ys = self.ys
        npairs = self.npairs
        a = np.zeros([self.n, npairs])
        aTs = np.zeros([npairs,1])

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last]/np.dot(y[:,last],y[:,last])
                q /= self.gamma

        for i in xrange(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                a[:,k] = y[:,k] - s[:,k]/self.gamma
                #print 'a:', a[:,k]
                #print 's:', s[:,k]
                for j in xrange(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        a[:,k] -= np.dot(a[:,l], s[:,k])/aTs[l] * a[:,l]
                aTs[k] = np.dot(a[:,k], s[:,k])
                #print 'a:', a[:,k]
                #print 'k:', k
                #print 'aTs:', aTs
                q += np.dot(a[:,k],v[:])/aTs[k]*a[:,k]
        return q

class LSR1_structured(LSR1):
    """
    LSR1 quasi newton using an unrolling formula.
    For this procedure see [Nocedal06]
    """
    def __init__(self, n, npairs=5, **kwargs):
        LSR1.__init__(self, n, npairs, **kwargs)
        self.yd = np.empty((self.n, self.npairs), 'd')
        self.accept_threshold = 1e-8

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
        a = np.zeros([self.n, npairs])
        ad = np.zeros([self.n, npairs])

        aTs = np.zeros([npairs,1])
        adTs = np.zeros([npairs,1])

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last]/np.dot(y[:,last],y[:,last])
                q /= self.gamma

        for i in xrange(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                a[:,k] = y[:,k] - s[:,k]/self.gamma
                ad[:,k] = yd[:,k] - s[:,k]/self.gamma
                for j in xrange(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        alTs = np.dot(a[:,l], s[:,k])
                        adlTs = np.dot(ad[:,l], s[:,k])
                        update = -alTs/aTs[l] * ad[:,l] - adlTs/aTs[l] * a[:,l] + adTs[l]/aTs[l] * alTs * a[:,l]
                        a[:,k] += update.copy()
                        ad[:,k] += update.copy()
                aTs[k] = np.dot(a[:,k], s[:,k])
                adTs[k] = np.dot(ad[:,k], s[:,k])
                aTv = np.dot(a[:,k],v[:])
                adTv = np.dot(ad[:,k],v[:])
                q += aTv/aTs[k] * ad[:,k] + adTv/aTs[k] * a[:,k] - aTv*adTs[k]/aTs[k]**2 * a[:,k]
        return q

    def store(self, new_s, new_y, new_yd):
        """
        Store the new pair (new_s,new_y). A new pair
        is only accepted if 
        | s_k' (y_k -B_k s_k) | >= 1e-8 ||s_k|| ||y_k - B_k s_k ||.
        """
        Bs = self.matvec(new_s)
        #print 'Bs:', Bs
        #print 'new_y:', new_y
        ymBs = new_yd - Bs
        criterion = abs(np.dot(ymBs, new_s)) >= self.accept_threshold * np.linalg.norm(new_s) * np.linalg.norm(ymBs)
        ymBsTs_criterion = abs(np.dot(ymBs, new_s)) >= 1e-15
        ys = np.dot(new_s, new_y)

        ys_criterion = True; scaling_criterion = True; yms_criterion = True
        if self.scaling:
            if abs(ys) >= 1e-15:
                scaling_factor = ys/np.dot(new_y, new_y)
                scaling_criterion = np.linalg.norm(new_y - new_s / scaling_factor) >= 1e-10
            else:
                ys_criterion = False
        else:
            if np.linalg.norm(new_y - new_s) < 1e-10:
                yms_criterion = False

        if ymBsTs_criterion and yms_criterion and scaling_criterion and criterion and ys_criterion:
            insert = self.insert
            self.s[:,insert] = new_s.copy()
            self.y[:,insert] = new_y.copy()
            self.yd[:,insert] = new_yd.copy()
            self.ys[insert] = ys
            self.insert += 1
            self.insert = self.insert % self.npairs
        else:
            self.log.debug('Not accepting LSR1 update: |<y-Bs,s>|= %s, y-s/gamma=%s, y-s = %s, ys=%s' % (criterion, scaling_criterion, yms_criterion, ys_criterion))
        return



class InverseLSR1(LSR1):
    """
    Class InverseLSR1 is similar to InverseLBFGS, except that it uses a different
    approximation scheme. Inheritance is currently taken through LSR1
    to avoid multiple-inheritance confusion.

    This class is useful in trust region methods, where the approximate Hessian
    is used in the model problem. LSR1 has the advantatge over LBFGS and LDFP
    of permitting approximations that are not positive-definite.
    """

    def __init__(self, n, npairs=5, **kwargs):
        LSR1.__init__(self, n, npairs, **kwargs)
        self.accept_threshold = 1e-8

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
        s = self.s ; y = self.y ; ys = self.ys
        npairs = self.npairs
        a = np.zeros(npairs,'d')
        minimat = np.zeros([npairs,npairs],'d')

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last]/np.dot(y[:,last],y[:,last])
                q *= self.gamma

        paircount = 0
        for i in xrange(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                a[k] = np.dot(s[:,k],v[:]) - np.dot(y[:,k],q[:])
                paircount += 1

        # Populate small matrix to be inverted
        for i in xrange(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                minimat[k,k] = -ys[k] - np.dot(y[:,k],y[:,k])*self.gamma
                for j in xrange(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        minimat[k,l] = np.dot(y[:,k],s[:,l]) - np.dot(y[:,k],y[:,l])*self.gamma
                        minimat[l,k] = minimat[k,l]

        if paircount > 0:
            rng = paircount
            b = np.linalg.solve(minimat[0:rng,0:rng],a[0:rng])

        for i in xrange(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                q += b[k]*s[:,k] - b[k]*self.gamma*y[:,k]

        return q


# end class



class LSR1_new(object):
    """
    Class LSR1_new is an experimental version that uses the unrolling formula 
    for the LSR1 approximation and a novel storage scheme to accelerate the 
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
        self.accept_threshold = 1.0e-8

        # Storage of the (s,y) pairs
        self.s = []
        self.y = []
        self.a = [None]*npairs     # Unrolled vectors for the matvec
        self.aTs = [None]*npairs   # Unrolled scalings for the matvec
        self.ys_new = 0.0
        self.yy_new = 0.0
        self.gamma = 1.0

        # Keep track of number of matrix-vector products.
        self.numMatVecs = 0

        logger_name = kwargs.get('logger_name', 'nlpy.lsr1')
        self.log = logging.getLogger(logger_name)
        #self.log.addHandler(logging.NullHandler())
        self.log.info('Logger created')


    def store(self, new_s, new_y):
        """
        Store the new pair (new_s,new_y). A new pair
        is only accepted if 
        | s_k' (y_k -B_k s_k) | >= 1e-8 ||s_k|| ||y_k - B_k s_k ||.
        """
        Bs = self.matvec(new_s)
        ymBs = new_y - Bs
        criterion = abs(np.dot(ymBs, new_s)) >= self.accept_threshold * np.linalg.norm(new_s) * np.linalg.norm(ymBs)
        ymBsTs_criterion = abs(np.dot(ymBs, new_s)) >= 1e-15
        ys = np.dot(new_s, new_y)
        yy = np.dot(new_y, new_y)

        ys_criterion = True; scaling_criterion = True; yms_criterion = True
        if self.scaling:
            if abs(ys) >= 1e-15:
                scaling_factor = ys/yy
                # scaling_criterion = np.linalg.norm(new_y - new_s / scaling_factor) >= 1e-10
            else:
                # ys_criterion = False
                scaling_factor = 1.0
            scaling_criterion = np.linalg.norm(new_y - new_s / scaling_factor) >= 1.e-10
        else:
            if np.linalg.norm(new_y - new_s) < 1e-10:
                yms_criterion = False

        if ymBsTs_criterion and yms_criterion and scaling_criterion and criterion and ys_criterion:
            self.s.append(new_s.copy())
            self.y.append(new_y.copy())
            self.ys_new = ys
            self.yy_new = yy
            if len(self.s) > self.npairs:
                del self.s[0]
                del self.y[0]
            else:
                self.stored_pairs += 1
            # end if

            # Recompute stored data for the matvec computation
            if self.scaling and abs(self.ys_new) >= 1e-15:
                self.gamma = self.ys_new / self.yy_new
            else:
                self.gamma = 1.0

            for i in xrange(self.stored_pairs):
                self.a[i] = self.y[i] - self.s[i]*self.diag/self.gamma
                for j in xrange(i):
                    self.a[i] -= np.dot(self.a[j], self.s[i])/self.aTs[j] * self.a[j]
                self.aTs[i] = np.dot(self.a[i], self.s[i])
            # end for
        else:
            self.log.debug('Not accepting LSR1 update: |<y-Bs,s>|= %s, y-s/gamma=%s, y-s = %s, ys=%s' % (criterion, scaling_criterion, yms_criterion, ys_criterion))
        return


    def restart(self):
        """
        Restart the approximation by clearing all data on past updates.
        """
        self.gamma = 1.0
        self.s = []
        self.y = []
        self.a = [None]*self.npairs
        self.aTs = [None]*self.npairs
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
            w += np.dot(self.a[i],v)/self.aTs[i] * self.a[i]
        # end for
        return w



class LSR1_structured_new(LSR1_new):
    """
    A structured LSR1 approximation based on the LSR1_new class.
    """
    def __init__(self, n, npairs=5, **kwargs):
        LSR1_new.__init__(self, n, npairs, **kwargs)
        self.yd = []
        self.ad = [None]*self.npairs
        self.adTs = [None]*self.npairs


    def store(self, new_s, new_y, new_yd):
        """
        Store the new pair (new_s,new_y). A new pair
        is only accepted if 
        | s_k' (y_k -B_k s_k) | >= 1e-8 ||s_k|| ||y_k - B_k s_k ||.
        """
        Bs = self.matvec(new_s)
        ymBs = new_yd - Bs
        criterion = abs(np.dot(ymBs, new_s)) >= self.accept_threshold * np.linalg.norm(new_s) * np.linalg.norm(ymBs)
        ymBsTs_criterion = abs(np.dot(ymBs, new_s)) >= 1e-15
        ys = np.dot(new_s, new_y)

        ys_criterion = True; scaling_criterion = True; yms_criterion = True
        if self.scaling:
            if abs(ys) >= 1e-15:
                scaling_factor = ys/np.dot(new_y, new_y)
                scaling_criterion = np.linalg.norm(new_y - new_s / scaling_factor) >= 1e-10
            else:
                ys_criterion = False
        else:
            if np.linalg.norm(new_y - new_s) < 1e-10:
                yms_criterion = False

        if ymBsTs_criterion and yms_criterion and scaling_criterion and criterion and ys_criterion:
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

            # Recompute stored data for the matvec computation
            if self.scaling:
                # ** This scaling criterion should probably change for the structured update
                self.gamma = ys / np.dot(new_y, new_y)

            for i in xrange(self.stored_pairs):
                self.a[i] = self.y[i] - self.s[i]/self.gamma
                self.ad[i] = self.yd[i] - self.s[i]/self.gamma
                for j in xrange(i):
                    aTs_temp = np.dot(self.a[j],self.s[i])
                    adTs_temp = np.dot(self.ad[j],self.s[i])
                    Delta_s = (aTs_temp/self.aTs[j])*self.ad[j] + (adTs_temp/self.aTs[j])*self.a[j]
                    Delta_s -= (aTs_temp*self.adTs[j]/self.aTs[j]**2)*self.a[j]
                    self.a[i] -= Delta_s
                    self.ad[i] -= Delta_s
                # end for
                self.aTs[i] = np.dot(self.a[i], self.s[i])
                self.adTs[i] = np.dot(self.ad[i], self.s[i])
            # end for
        else:
            self.log.debug('Not accepting LSR1 update: |<y-Bs,s>|= %s, y-s/gamma=%s, y-s = %s, ys=%s' % (criterion, scaling_criterion, yms_criterion, ys_criterion))
        return


    def restart(self):
        """
        Restart the approximation by clearing all data on past updates.
        """
        LSR1_new.restart(self)
        self.yd = []
        self.ad = [None]*self.npairs
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
            aTv = np.dot(self.a[i],v)
            adTv = np.dot(self.ad[i],v)
            w += (aTv/self.aTs[i])*self.ad[i] + (adTv/self.aTs[i])*self.a[i]
            w -= (aTv*self.adTs[i]/self.aTs[i]**2)*self.a[i]
        # end for
        return w



class LSR1_infeas(LSR1_new):
    """
    This is a specialized LSR1 approximation for the infeasibility term of 
    an augmented Lagrangian function. The class contains specialized methods 
    for defining an initial diagonal.
    """

    def __init__(self, n, x, vecfunc, jprod, jtprod, npairs=5, **kwargs):
        LSR1_new.__init__(self, n, npairs, **kwargs)
        self.slack_index = kwargs.get('slack_index',n)
        self.jprod = jprod
        self.jtprod = jtprod
        self.x = x  # Point at which to compute diagonal
        # self.compute_diag()
        self.diag_eps = 1e-6
        self.beta = kwargs.get('beta',min(self.slack_index,3))
        dummy = vecfunc(self.x) # A call to the constraint function to set up the jprod internals
        self.compute_diag()


    def restart(self, x):
        """
        Restart the approximation.
        """
        LSR1_new.restart(self)
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
