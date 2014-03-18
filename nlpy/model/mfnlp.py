"""
mfnlp.py

An abstract class of a matrix-free NLP model for NLPy.
"""


__docformat__ = 'restructuredtext'

# =============================================================================
# Standard Python modules
# =============================================================================
import copy
import pdb
# =============================================================================
# External Python modules
# =============================================================================
import numpy as np
from numpy.linalg import norm
eps = np.finfo(1.0).eps

# =============================================================================
# Extension modules
# =============================================================================
from nlpy.model import NLPModel
from nlpy.krylov import SimpleLinearOperator
from nlpy.tools import List


class MFModel(NLPModel):

    """
    MFModel is a derived type of NLPModel which focuses on matrix-free
    implementations of the standard NLP.

    Most of the functionality is the same as NLPModel except for additional
    methods and counters for Jacobian-free cases.

    Note: these parts could be reintegrated into NLPModel at a later date.
    """

    def __init__(self, n=0, m=0, name='Generic Matrix-Free', **kwargs):

        # Standard NLP initialization
        NLPModel.__init__(self,n=n,m=m,name=name,**kwargs)


    def jac(self, x, **kwargs):
        return SimpleLinearOperator(self.n, self.m, symmetric=False,
                         matvec=lambda u: self.jprod(x,u,**kwargs),
                         matvec_transp=lambda u: self.jtprod(x,u,**kwargs))


    def hess(self, x, z=None, **kwargs):
        return SimpleLinearOperator(self.n, self.n, symmetric=True,
                         matvec=lambda u: self.hprod(x,z,u,**kwargs))
# end class



class SlackNLP( MFModel ):
    """
    General framework for converting a nonlinear optimization problem to a
    form using slack variables.

    In the latter problem, the only inequality constraints are bounds on
    the slack variables. The other constraints are (typically) nonlinear
    equalities.

    The order of variables in the transformed problem is as follows:

    1. x, the original problem variables.

    2. sL = [ sLL | sLR ], sLL being the slack variables corresponding to
       general constraints with a lower bound only, and sLR being the slack
       variables corresponding to the 'lower' side of range constraints.

    3. sU = [ sUU | sUR ], sUU being the slack variables corresponding to
       general constraints with an upper bound only, and sUR being the slack
       variables corresponding to the 'upper' side of range constraints.

    4. tL = [ tLL | tLR ], tLL being the slack variables corresponding to
       variables with a lower bound only, and tLR being the slack variables
       corresponding to the 'lower' side of two-sided bounds.

    5. tU = [ tUU | tUR ], tUU being the slack variables corresponding to
       variables with an upper bound only, and tLR being the slack variables
       corresponding to the 'upper' side of two-sided bounds.

    This framework initializes the slack variables sL, sU, tL, and tU to
    zero by default.

    Note that the slack framework does not update all members of AmplModel,
    such as the index set of constraints with an upper bound, etc., but
    rather performs the evaluations of the constraints for the updated
    model implicitly.
    """

    def __init__(self, nlp, keep_variable_bounds=False, **kwargs):

        self.nlp = nlp
        self.keep_variable_bounds = keep_variable_bounds

        # Save number of variables and constraints prior to transformation
        self.original_n = nlp.n
        self.original_m = nlp.m

        # Number of slacks for inequality constraints with a lower bound
        n_con_low = nlp.nlowerC + nlp.nrangeC ; self.n_con_low = n_con_low

        # Number of slacks for inequality constraints with an upper bound
        n_con_up = nlp.nupperC + nlp.nrangeC ; self.n_con_up = n_con_up

        # Number of slacks for variables with a lower bound
        n_var_low = nlp.nlowerB + nlp.nrangeB ; self.n_var_low = n_var_low

        # Number of slacks for variables with an upper bound
        n_var_up = nlp.nupperB + nlp.nrangeB ; self.n_var_up = n_var_up

        # Update effective number of variables and constraints
        if keep_variable_bounds==True:
            n = self.original_n + n_con_low + n_con_up
            m = self.original_m + nlp.nrangeC

            Lvar = np.zeros(n)
            Lvar[:self.original_n] = nlp.Lvar
            Uvar = +np.infty * np.ones(n)
            Uvar[:self.original_n] = nlp.Uvar

        else:
            n = self.original_n + n_con_low + n_con_up + n_var_low + n_var_up
            m = self.original_m + nlp.nrangeC + n_var_low + n_var_up
            Lvar = np.zeros(n)
            Uvar = +np.infty * np.ones(n)

        Lcon = Ucon = np.zeros(m)

        MFModel.__init__(self, n=n, m=m, name='Slack NLP', Lvar=Lvar, \
                          Uvar=Uvar, Lcon=Lcon, Ucon=Ucon)

        self.hprod = nlp.hprod
        self.hiprod = self.hiprod

        self.equalC = np.array(nlp.equalC, dtype='int64') ; self.nequalC = nlp.nequalC
        self.lowerC = np.array(nlp.lowerC, dtype='int64') ; self.nlowerC = nlp.nlowerC
        self.upperC = np.array(nlp.upperC, dtype='int64') ; self.nupperC = nlp.nupperC
        self.rangeC = np.array(nlp.rangeC, dtype='int64') ; self.nrangeC = nlp.nrangeC

        # Redefine primal and dual initial guesses
        self.original_x0 = nlp.x0[:]
        self.x0 = np.zeros(self.n)
        self.x0[:self.original_n] = self.original_x0[:]

        self.original_pi0 = nlp.pi0[:]
        self.pi0 = np.zeros(self.m)
        self.pi0[:self.original_m] = self.original_pi0[:]

        # Saved values (private).
        # No more hashing objects to check x: 
        # Numpy norm calculation is much faster for long arrays
        self._cache = {'x':np.infty * np.ones(self.original_n,'d'),
            'obj':None, 'cons':None, 'grad':None}
        # self._last_x = np.infty * np.ones(self.original_n,'d')
        # self._last_x_hash = hashlib.sha1(self._last_x).hexdigest()
        # self._last_obj = None
        # self._last_cons = None
        # self._last_grad = None

        return


    def InitializeSlacks(self, val=0.0, **kwargs):
        """
        Initialize all slack variables to given value. This method may need to
        be overridden.
        """
        self.x0[self.original_n:] = val
        return


    def obj(self, x):
        """
        Return the value of the objective function at `x`. This function is
        specialized since the original objective function only depends on a
        subvector of `x`.
        """

        # x_hash = hashlib.sha1(x[:self.original_n]).hexdigest()
        # same_x = self._last_x_hash == x_hash
        # same_x = (self._last_x == x[:self.original_n]).all()

        # pdb.set_trace()
        same_x = norm(x[:self.original_n] - self._cache['x']) < eps

        # if self._cache['obj'] is not None and same_x:
        #     f = self._cache['obj']
        # elif self._cache['obj'] is None and same_x:
        #     f = self.nlp.obj(self._cache['x'])
        #     self._cache['obj'] = copy.deepcopy(f)
        # else:
        #     f = self.nlp.obj(x[:self.original_n])
        #     self._cache['x'] = x[:self.original_n].copy()
        #     self._cache['obj'] = copy.deepcopy(f)
        #     self._cache['cons'] = None
        #     self._cache['grad'] = None

        # Debug mode
        f = self.nlp.obj(x[:self.original_n])

        return f


    def grad(self, x):
        """
        Return the value of the gradient of the objective function at `x`.
        This function is specialized since the original objective function only
        depends on a subvector of `x`.
        """
        g = np.zeros(self.n)
        same_x = norm(x[:self.original_n] - self._cache['x']) < eps

        # x_hash = hashlib.sha1(x[:self.original_n]).hexdigest()
        # same_x = self._last_x_hash == x_hash
        # same_x = (self._last_x == x[:self.original_n]).all()

        # if self._cache['grad'] is not None and same_x:
        #     g[:self.original_n] = self._cache['grad']
        # elif self._cache['grad'] is None and same_x:
        #     g[:self.original_n] = self.nlp.grad(self._cache['x'])
        #     self._cache['grad'] = copy.deepcopy(g[:self.original_n])
        # else:
        #     g[:self.original_n] = self.nlp.grad(x[:self.original_n])
        #     self._cache['x'] = x[:self.original_n].copy()
        #     self._cache['obj'] = None
        #     self._cache['cons'] = None
        #     self._cache['grad'] = copy.deepcopy(g[:self.original_n])

        # Debug mode
        g[:self.original_n] = self.nlp.grad(x[:self.original_n])

        return g


    def cons(self, x):
        """
        Evaluate the vector of general constraints for the modified problem.
        Constraints are stored in the order in which they appear in the
        original problem. If constraint i is a range constraint, c[i] will
        be the constraint that has the slack on the lower bound on c[i].
        The constraint with the slack on the upper bound on c[i] will be stored
        in position m + k, where k is the position of index i in
        rangeC, i.e., k=0 iff constraint i is the range constraint that
        appears first, k=1 iff it appears second, etc.

        Constraints appear in the following order:

        1. [ c  ]   general constraints in original order
        2. [ cR ]   'upper' side of range constraints
        3. [ b  ]   linear constraints corresponding to bounds on original problem
        4. [ bR ]   linear constraints corresponding to 'upper' side of two-sided
                    bounds
        """
        n = self.n ; on = self.original_n
        m = self.m ; om = self.original_m
        nlp = self.nlp

        equalC = self.equalC
        lowerC = self.lowerC ; nlowerC = self.nlowerC
        upperC = self.upperC ; nupperC = self.nupperC
        rangeC = self.rangeC ; nrangeC = self.nrangeC

        mslow = on + self.n_con_low
        msup  = mslow + self.n_con_up
        s_low = x[on:mslow]    # len(s_low) = n_con_low
        s_up  = x[mslow:msup]  # len(s_up)  = n_con_up

        c = np.empty(m)
        same_x = norm(x[:on] - self._cache['x']) < eps

        # x_hash = hashlib.sha1(x[:self.original_n]).hexdigest()
        # same_x = self._last_x_hash == x_hash
        # same_x = (self._last_x == x[:self.original_n]).all()

        # if self._cache['cons'] is not None and same_x:
        #     c[:om] = self._cache['cons']
        # elif self._cache['cons'] is None and same_x:
        #     c[:om] = self.nlp.cons(self._cache['x'])
        #     self._cache['cons'] = copy.deepcopy(c[:om])
        # else:
        #     c[:om] = self.nlp.cons(x[:on])
        #     self._cache['x'] = x[:on]
        #     self._cache['obj'] = None
        #     self._cache['cons'] = copy.deepcopy(c[:om])
        #     self._cache['grad'] = None

        # Debug mode
        c[:om] = self.nlp.cons(x[:on])

        c[om:om+nrangeC] = c[rangeC]

        c[equalC] -= nlp.Lcon[equalC]
        c[lowerC] -= nlp.Lcon[lowerC] ; c[lowerC] -= s_low[:nlowerC]

        c[upperC] -= nlp.Ucon[upperC] ; c[upperC] *= -1.
        c[upperC] -= s_up[:nupperC]

        c[rangeC] -= nlp.Lcon[rangeC] ; c[rangeC] -= s_low[nlowerC:]

        c[om:om+nrangeC] -= nlp.Ucon[rangeC]
        c[om:om+nrangeC] *= -1
        c[om:om+nrangeC] -= s_up[nupperC:]

        if self.keep_variable_bounds==False:
            # Add linear constraints corresponding to bounds on original problem
            lowerB = nlp.lowerB ; nlowerB = nlp.nlowerB ; Lvar = nlp.Lvar
            upperB = nlp.upperB ; nupperB = nlp.nupperB ; Uvar = nlp.Uvar
            rangeB = nlp.rangeB ; nrangeB = nlp.nrangeB

            nt = on + self.n_con_low + self.n_con_up
            ntlow = nt + self.n_var_low
            t_low = x[nt:ntlow]
            t_up  = x[ntlow:]

            b = c[om+nrangeC:]

            b[:nlowerB] = x[lowerB] - Lvar[lowerB] - t_low[:nlowerB]
            b[nlowerB:nlowerB+nrangeB] = x[rangeB] - Lvar[rangeB] \
                                        - t_low[nlowerB:]
            b[nlowerB+nrangeB:nlowerB+nrangeB+nupperB] = Uvar[upperB] \
                                        - x[upperB] - t_up[:nupperB]
            b[nlowerB+nrangeB+nupperB:] = Uvar[rangeB] - x[rangeB] \
                                        - t_up[nupperB:]

        return c


    def Bounds(self, x):
        """
        Evaluate the vector of equality constraints corresponding to bounds
        on the variables in the original problem.
        """
        lowerB = self.lowerB ; nlowerB = self.nlowerB
        upperB = self.upperB ; nupperB = self.nupperB
        rangeB = self.rangeB ; nrangeB = self.nrangeB

        n  = self.n ; on = self.original_n
        mslow = on + nrangeC + self.n_con_low
        msup  = mslow + self.n_con_up
        nt = self.original_n + self.n_con_low + self.n_con_up
        ntlow = nt + self.n_var_low

        t_low  = x[msup:ntlow]
        t_up   = x[ntlow:]

        b = np.empty(n + nrangeB)
        b[:n] = x[:]
        b[n:] = x[rangeB]

        b[lowerB] -= self.Lvar[lowerB] ; b[lowerB] -= t_low[:nlowerB]

        b[upperB] -= self.Uvar[upperB] ; b[upperB] *= -1
        b[upperB] -= t_up[:nupperB]

        b[rangeB] -= self.Lvar[rangeB] ; b[rangeB] -= t_low[nlowerB:]
        b[n:]     -= self.Uvar[rangeB] ; b[n:] *= -1
        b[n:]     -= t_up[nupperB:]

        return b


    def jprod(self, x, v, **kwargs):

        nlp = self.nlp
        on = self.original_n
        om = self.original_m
        n = self.n
        m = self.m

        lowerC = self.lowerC ; nlowerC = self.nlowerC
        upperC = self.upperC ; nupperC = self.nupperC
        rangeC = self.rangeC ; nrangeC = self.nrangeC
        lowerB = nlp.lowerB ; nlowerB = nlp.nlowerB
        upperB = nlp.upperB ; nupperB = nlp.nupperB
        rangeB = nlp.rangeB ; nrangeB = nlp.nrangeB
        nbnds  = nlowerB + nupperB + 2*nrangeB
        nSlacks = nlowerC + nupperC + 2*nrangeC

        p = np.zeros(m)

        # Perform jprod and account for upper bounded constraints
        if kwargs.get('sparse_only',False) == False:
            p[:om] = nlp.jprod(x[:on], v[:on], **kwargs)
        # (Otherwise, do not call the function - assume constraint set is dense) 

        p[upperC] *= -1.0
        p[om:om+nrangeC] = p[rangeC]
        p[om:om+nrangeC] *= -1.0

        # Insert contribution of slacks on general constraints
        bot = on;        p[lowerC] -= v[bot:bot+nlowerC]
        bot += nlowerC;  p[rangeC] -= v[bot:bot+nrangeC]
        bot += nrangeC;  p[upperC] -= v[bot:bot+nupperC]
        bot += nupperC;  p[om:om+nrangeC] -= v[bot:bot+nrangeC]

        if self.keep_variable_bounds==False:
            # Insert contribution of bound constraints on the original problem
            bot  = m+nrangeC ;

            # To be finished

        return p


    def jtprod(self, x, v, **kwargs):

        nlp = self.nlp
        on = self.original_n
        om = self.original_m
        n = self.n
        m = self.m

        # List() simply allows operations such as 1 + [2,3] -> [3,4]
        lowerC = self.lowerC ; nlowerC = self.nlowerC
        upperC = self.upperC ; nupperC = self.nupperC
        rangeC = self.rangeC ; nrangeC = self.nrangeC
        lowerB = nlp.lowerB ; nlowerB = nlp.nlowerB
        upperB = nlp.upperB ; nupperB = nlp.nupperB
        rangeB = nlp.rangeB ; nrangeB = nlp.nrangeB
        nbnds  = nlowerB + nupperB + 2*nrangeB
        nSlacks = nlowerC + nupperC + 2*nrangeC

        p = np.zeros(n)
        vmp = v[:om].copy()
        vmp[upperC] *= -1.0
        vmp[rangeC] -= v[om:]

        # Perform jtprod and account for upper bounded constraints
        if kwargs.get('sparse_only',False) == False:
            p[:on] = nlp.jtprod(x[:on], vmp, **kwargs)
        # (Otherwise, do not call the function - assume constraint set is dense) 

        # Insert contribution of slacks on general constraints
        bot = on;       p[on:on+nlowerC]    = -v[lowerC]
        bot += nlowerC; p[bot:bot+nrangeC]  = -v[rangeC]
        bot += nrangeC; p[bot:bot+nupperC]  = -v[upperC]
        bot += nupperC; p[bot:bot+nrangeC]  = -v[om:om+nrangeC]

        if self.keep_variable_bounds==False:
            # Insert contribution of bound constraints on the original problem
            bot  = m+nrangeC ;

            # To be finished

        return p


