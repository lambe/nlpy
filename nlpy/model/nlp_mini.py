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
from nlpy.model import NLPModel

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
        self.Infinity = 1e+20
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

        # Initialize some counters
        self.feval = 0    # evaluations of objective  function
        self.geval = 0    #                           gradient
        self.Heval = 0    #                Lagrangian Hessian
        self.Hprod = 0    #                matrix-vector products with Hessian
        self.ceval = 0    #                constraint functions
        self.Jeval = 0    #                           gradients
        self.Jprod = 0    #                matrix-vector products with Jacobian
        self.JTprod = 0   #                             with transpose Jacobian



# New matrix-free class and SlackNLP class go here