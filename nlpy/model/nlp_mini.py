"""
nlp_mini.py

A development version of the NLPModel class. Instead of using index lists to 
define which constraints are of which type, we force the user into adopting a 
specific order. The order is as follows:

nonlinear equalities (=)
nonlinear lower-bound inequalities (>=)
nonlinear range inequalities (c_L <= c <= c_U)
nonlinear upper-bound inequalities (<=)

Linear and other special constraint forms are not implemented here. This 
should speed things up considerably, especially for the SlackNLP constraint 
handling.

A corresponding SlackNLP derived class has also been implemented here.

Last update: March 2014
"""

import numpy as np 
import copy
from numpy.linalg import norm
from nlpy.model import NLPModel
eps = np.finfo(1.0).eps

class NLPModel_mini(NLPModel):
	"""
	A derived NLPModel class with a simplified constraint definition. Only 
	nonlinear constraints are assumed to be present in this formulation.

	We assume the constraint order is equalities, lower-bounded inequalities, 
	range inequalities, and upper-bounded inequalities. Bad things will happen 
	if this order is violated.

    :parameters:

        :n:       number of variables (default: 0)
        :m:       number of general (non bound) constraints (default: 0)
        :name:    model name (default: 'Generic')

    :keywords:

        :x0:      initial point (default: all 0)
        :pi0:     vector of initial multipliers (default: all 0)
        :m_list:  vector of length 4 defining the number of constraints 
        		  in each group (automatic detection enabled if this is 
        		  not given)
        :Lvar:    vector of lower bounds on the variables
                  (default: all -Infinity)
        :Uvar:    vector of upper bounds on the variables
                  (default: all +Infinity)
        :Lcon:    vector of lower bounds on the constraints
                  (default: all -Infinity)
        :Ucon:    vector of upper bounds on the constraints
                  (default: all +Infinity)

	"""

	# Just need to write the class constructor
    def __init__(self, n=0, m=0, name='Generic', **kwargs):

        self.nvar = self.n = n   # Number of variables
        self.ncon = self.m = m   # Number of general constraints
        self.name = name         # Problem name

        # Initialize local value for Infinity
        self.Infinity = np.infty
        self.negInfinity = - self.Infinity
        self.zero = 0.0
        self.one = 1.0

        # Set initial point
        if 'x0' in kwargs.keys():
            self.x0 = np.ascontiguousarray(kwargs['x0'], dtype=float)
        else:
            self.x0 = np.zeros(self.n, 'd')

        # Set initial multipliers
        if 'pi0' in kwargs.keys():
            self.pi0 = np.ascontiguousarray(kwargs['pi0'], dtype=float)
        else:
            self.pi0 = np.zeros(self.m, 'd')

        # Set lower bounds on variables    Lvar[i] <= x[i]  i = 1,...,n
        if 'Lvar' in kwargs.keys():
            self.Lvar = np.ascontiguousarray(kwargs['Lvar'], dtype=float)
        else:
            self.Lvar = self.negInfinity * np.ones(self.n, 'd')

        # Set upper bounds on variables    x[i] <= Uvar[i]  i = 1,...,n
        if 'Uvar' in kwargs.keys():
            self.Uvar = np.ascontiguousarray(kwargs['Uvar'], dtype=float)
        else:
            self.Uvar = self.Infinity * np.ones(self.n, 'd')

        # Set lower bounds on constraints  Lcon[i] <= c[i]  i = 1,...,m
        if 'Lcon' in kwargs.keys():
            self.Lcon = np.ascontiguousarray(kwargs['Lcon'], dtype=float)
        else:
            self.Lcon = self.negInfinity * np.ones(self.m, 'd')

        # Set upper bounds on constraints  c[i] <= Ucon[i]  i = 1,...,m
        if 'Ucon' in kwargs.keys():
            self.Ucon = np.ascontiguousarray(kwargs['Ucon'], dtype=float)
        else:
            self.Ucon = self.Infinity * np.ones(self.m, 'd')

        # Count the number of constraints of each type
        if 'm_list' in kwargs.keys():
        	# User-provided values
        	m_list = np.ascontiguousarray(kwargs['m_list'], dtype=int)
        	self.nequalC = m_list[0]
        	self.nlowerC = m_list[1]
        	self.nrangeC = m_list[2]
        	self.nupperC = m_list[3]

        	self.equalC_start = 0
        	self.lowerC_start = m_list[0]
        	self.rangeC_start = m_list[0] + m_list[1]
        	self.upperC_start = m_list[0] + m_list[1] + m_list[2]

        	# Sanity check of constraint bounds (to be implemented)
        	if self.m != m_list.sum():
        		raise ValueError('Number of constraints does not match provided list')

        else:
        	# Auto detection of number of constraints in each class
        	k = 0
        	self.equalC_start = 0
        	while self.Lcon[k] == self.Ucon[k]:
        		k += 1
        	self.nequalC = k
        	ktotal = k

        	self.lowerC_start = ktotal
        	while self.Lcon[k] > self.negInfinity and self.Ucon[k] == self.Infinity:
        		k += 1
        	self.nlowerC = k - ktotal
        	ktotal += k

        	self.rangeC_start = ktotal
        	while self.Lcon[k] > self.negInfinity and self.Ucon[k] < self.Infinity:
        		k += 1
        	self.nrangeC = k - ktotal
        	ktotal += k

        	self.upperC_start = ktotal
        	while self.Lcon[k] == self.negInfinity and self.Ucon[k] < self.Infinity:
        		k += 1
        	self.nupperC = k - ktotal
        	ktotal += k

        	if ktotal != self.m:
        		raise ValueError('Constraint bounds badly ordered or free constraints present')

        # Same index values as defined above, shorthand
        self.neC = self.nequalC
        self.nlC = self.nlowerC
        self.nrC = self.nrangeC
        self.nuC = self.nupperC

        self.eCs = self.equalC_start
        self.lCs = self.lowerC_start
        self.rCs = self.rangeC_start
        self.uCs = self.upperC_start

        # Initialize some counters
        self.feval = 0    # evaluations of objective  function
        self.geval = 0    #                           gradient
        self.Heval = 0    #                Lagrangian Hessian
        self.Hprod = 0    #                matrix-vector products with Hessian
        self.ceval = 0    #                constraint functions
        self.Jeval = 0    #                           gradients
        self.Jprod = 0    #                matrix-vector products with Jacobian
        self.JTprod = 0   #                             with transpose Jacobian



class MFModel_mini(NLPModel_mini):

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



# Is class inheritance necessary? Or do we just need something with 
# appropriate function names?
class SlackNLP_mini( MFModel_mini ):
    """
    General framework for converting a nonlinear optimization problem to a
    form using slack variables. Inequality constraints are transformed to 
    equalities with slack variables. The slack variables are then added 
    to the variable set and treated as bounded variables.

    The order of variables in the transformed problem is as follows:

    1. x, the original problem variables.

    2. sL = [ sLL | sLR ], sLL being the slack variables corresponding to
       general constraints with a lower bound only, and sLR being the slack
       variables corresponding to the 'lower' side of range constraints.

    3. sU = [ sUU | sUR ], sUU being the slack variables corresponding to
       general constraints with an upper bound only, and sUR being the slack
       variables corresponding to the 'upper' side of range constraints.

    This framework initializes the slack variables sL and sU to zero by 
    default.
    """

    def __init__(self, nlp, **kwargs):

        self.nlp = nlp

        # Save number of variables and constraints prior to transformation
        self.original_n = nlp.n
        self.original_m = nlp.m
        self.on = self.original_n
        self.om = self.original_m

        # Number of slacks for inequality constraints with a lower bound
        self.n_con_low = nlp.nlowerC + nlp.nrangeC

        # Number of slacks for inequality constraints with an upper bound
        self.n_con_up = nlp.nupperC + nlp.nrangeC

        # Update effective number of variables and constraints
        n = self.original_n + n_con_low + n_con_up
        m = self.original_m + nlp.nrangeC

        Lvar = np.zeros(n)
        Lvar[:self.original_n] = nlp.Lvar
        Uvar = +np.infty * np.ones(n)
        Uvar[:self.original_n] = nlp.Uvar

        Lcon = Ucon = np.zeros(m)

        MFModel.__init__(self, n=n, m=m, name='Slack NLP', Lvar=Lvar, \
                          Uvar=Uvar, Lcon=Lcon, Ucon=Ucon)

        self.hprod = nlp.hprod
        self.hiprod = nlp.hiprod

        # Redefine primal and dual initial guesses
        self.original_x0 = nlp.x0[:]
        self.x0 = np.zeros(self.n)
        self.x0[:self.original_n] = self.original_x0[:]

        self.original_pi0 = nlp.pi0[:]
        self.pi0 = np.zeros(self.m)
        self.pi0[:self.original_m] = self.original_pi0[:]

        # Saved values (private) 
        # No more hashing objects to check x: 
        # Numpy norm calculation is much faster for long arrays
        self._cache = {'x':np.infty * np.ones(self.original_n,'d'),
            'obj':None, 'cons':None, 'grad':None}

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

        same_x = norm(x[:self.original_n] - self._cache['x']) < eps

        if self._last_obj is not None and same_x:
            f = self._cache['obj']
        elif self._last_obj is None and same_x:
            f = self.nlp.obj(self._cache['x'])
            self._cache['obj'] = copy.deepcopy(f)
        else:
            f = self.nlp.obj(x[:self.original_n])
            self._cache['x'] = x[:self.original_n].copy()
            self._cache['obj'] = copy.deepcopy(f)
            self._cache['cons'] = None
            self._cache['grad'] = None

        return f


    def grad(self, x):
        """
        Return the value of the gradient of the objective function at `x`.
        This function is specialized since the original objective function only
        depends on a subvector of `x`.
        """
        g = np.zeros(self.n)
        same_x = norm(x[:self.original_n] - self._cache['x']) < eps

        if self._cache['grad'] is not None and same_x:
            g[:self.original_n] = self._cache['grad']
        elif self._cache['grad'] is None and same_x:
            g[:self.original_n] = self.nlp.grad(self._cache['x'])
            self._cache['grad'] = copy.deepcopy(g[:self.original_n])
        else:
            g[:self.original_n] = self.nlp.grad(x[:self.original_n])
            self._cache['x'] = x[:self.original_n].copy()
            self._cache['obj'] = None
            self._cache['cons'] = None
            self._cache['grad'] = copy.deepcopy(g[:self.original_n])

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
        """
        mslow = self.original_n + self.n_con_low
        msup  = mslow + self.n_con_up
        s_low = x[on:mslow]    # len(s_low) = n_con_low
        s_up  = x[mslow:msup]  # len(s_up)  = n_con_up

        c = np.empty(self.m)
        same_x = norm(x[:self.original_n] - self._cache['x']) < eps

        if self._cache['cons'] is not None and same_x:
            c[:self.original_m] = self._cache['cons']
        elif self._cache['cons'] is None and same_x:
            c[:self.original_m] = self.nlp.cons(self._cache['x'])
            self._cache['cons'] = copy.deepcopy(c[:self.original_m])
        else:
            c[:self.original_m] = self.nlp.cons(x[:self.original_n])
            self._cache['x'] = x[:self.original_n]
            self._cache['obj'] = None
            self._cache['cons'] = copy.deepcopy(c[:self.original_m])
            self._cache['grad'] = None

        # Copy range constraint evaluation
        c[self.om:] = c[self.nlp.rCs:self.nlp.uCs]

        # Equality constraints
        c[self.nlp.eCs:self.nlp.lCs] -= self.nlp.Lcon[self.nlp.eCs:self.nlp.lCs]

        # Lower bounded constraints
        c[self.nlp.lCs:self.nlp.rCs] -= self.nlp.Lcon[self.nlp.lCs:self.nlp.rCs] 
        c[self.nlp.lCs:self.nlp.rCs] -= s_low[:self.nlp.nlC]

        # Lower bound of range constraints
        c[self.nlp.rCs:self.nlp.uCs] -= self.nlp.Lcon[self.nlp.rCs:self.nlp.uCs] 
        c[self.nlp.rCs:self.nlp.uCs] -= s_low[self.nlp.nlC:]

        # Upper bounded constraints
        c[self.nlp.uCs:self.om] -= self.nlp.Ucon[self.nlp.uCs:self.om] 
        c[self.nlp.uCs:self.om] *= -1.
        c[self.nlp.uCs:self.om] -= s_up[:self.nlp.nuC]

        # Upper bound of range constraints
        c[self.om:] -= self.nlp.Ucon[self.nlp.rCs:self.nlp.uCs]
        c[self.om:] *= -1
        c[self.om:] -= s_up[self.nlp.nuC:]

        return c


    def jprod(self, x, v, **kwargs):

        p = np.zeros(m)

        # Perform jprod and account for upper bounded constraints
        p[:self.om] = nlp.jprod(x[:self.on], v[:self.on], **kwargs)
        p[self.nlp.uCs:self.om] *= -1.0
        p[self.om:] = p[self.nlp.rCs:self.nlp.uCs]
        p[self.om:] *= -1.0

        # Insert contribution of slacks on general constraints
        bot = self.on           # Lower bound
        p[self.nlp.lCs:self.nlp.rCs] -= v[bot:bot+self.nlp.nlC]
        bot += self.nlp.nlC     # Lower range
        p[self.nlp.rCs:self.nlp.uCs] -= v[bot:bot+self.nlp.nrC]
        bot += self.nlp.nrC     # Upper bound
        p[self.nlp.uCs:self.om] -= v[bot:bot+self.nlp.nuC]
        bot += self.nlp.nuC     # Upper range
        p[self.om:] -= v[bot:bot+self.nlp.nrC]

        return p


    def jtprod(self, x, v, **kwargs):

        p = np.zeros(n)
        vmp = v[:self.om].copy()
        vmp[self.nlp.uCs:self.om] *= -1.0
        vmp[self.nlp.rCs:self.nlp.uCs] -= v[self.om:]

        p[:self.on] = nlp.jtprod(x[:self.on], vmp, **kwargs)

        # Insert contribution of slacks on general constraints
        bot = self.on           # Lower bound
        p[bot:bot+self.nlp.nlC] = -v[self.nlp.lCs:self.nlp.rCs]
        bot += self.nlp.nlC     # Lower range
        p[bot:bot+self.nlp.nrC] = -v[self.nlp.rCs:self.nlp.uCs]
        bot += self.nlp.nrC     # Upper bound
        p[bot:bot+self.nlp.nuC]  = -v[self.nlp.uCs:self.om]
        bot += nupperC          # Upper range
        p[bot:bot+self.nlp.nrC]  = -v[self.om:]

        return p


