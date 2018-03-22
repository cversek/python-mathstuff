###############################################################################
# optimizer.py
# desc: interfaces for cost function optimization:
#       NLSOptimize - scipy's Levenberg-Marquardt algorithm 
#                     scipy.optimize.leastsq
# authors: Craig Versek (cversek@physics.umass.edu)
#          Mike Thorn
###############################################################################
from enthought.traits.api import HasTraits, on_trait_change, Str, Int, Float, \
                                 Bool, Tuple, List, Dict, Undefined, Array,   \
                                 Function, Instance
from enthought.traits.ui.api import View, Item, Group
import numpy, scipy
import scipy.optimize
###############################################################################
class OptimizerBase(HasTraits):
    cost_map       = Function                 #function taking N parameters and yielding M values
    args           = List                     #additional arguments to the cost function 
    P              = Array(dtype = numpy.float64, shape=(None,))     #1D array, represents a set of N parameters with ordering   
    cost           = Array(dtype = numpy.float64, shape=(None,))     #1D array, length M
    ndf            = Int                      #number of degrees of freedom
 
    def __init__(self, cost_map, P0, args = None):
        super(OptimizerBase, self).__init__()
        self.cost_map  = cost_map
        if args is None:
            args = []
        self.args      = list(args)
        self.P         = P0
        
    @on_trait_change('P')
    def update_cost(self):
        self.cost = self.cost_map(self.P,*self.args)
        self.ndf  = len(self.P)

    def optimize(self, P0 = None, args = None):
        raise NotImplementedError

###############################################################################
class NLSOptimizer(OptimizerBase):
    """ A traits enabled class that wraps scipy.optimize.leastsq, a routine:
        "which minimizes the sum of squares of M (non-linear) equations 
         in N unknowns given a starting estimate, x0, using a modification of 
         the Levenberg-Marquardt algorithm"
    """
    covar          = Array(dtype = numpy.float64, shape=(None,None)) #2D array, length NxN    
    success        = Bool  
    infodict       = Dict
    message        = Str
    ier            = Int

    view = View( Item('P'    ,   label = 'Parameter Values'),
                 Item('covar',   label = 'Parameter Covariance'),
                 Item('success'),
                 Item('infodict'),
                 Item('message'),
                 Item('ier'),
                 height    = 0.75,
                 width     = 0.25,
                 resizable = True,               
               )

    def optimize(self, 
                 P0 = None, 
                 args = None, 
                 steps = None, #FIXME don't use this parameter, resulsts are unpredictable
                ):

        if not P0 is None:
            self.P = P0
        if not args is None:
            self.args = list(args)
        
        P0   = self.P[:]
        N    = len(P0)
        args = tuple(self.args)

        #compute the number of times to evaluate the cost_map function
        maxfev = 0 #scipy default = 100*(N + 1)
        if not steps is None:
            steps  = int(steps)
            maxfev = steps*(N + 1) #each step requires N + 1 evaulations
        
        if N > 0:  #cannot be an empty sequence
            results = scipy.optimize.leastsq( self.cost_map, 
                                              P0,
                                              args = args, 
                                              full_output = True,
                                              maxfev = maxfev,
                                              #factor=1.0
                                            )
            #unpack the results
            new_P, covar, infodict, mesg, ier = results
            #convert and store all the data
            self.infodict = infodict
            self.message  = mesg
            self.ier      = ier
            #condition for success
            if ier in (1,2,3,4):
                self.success  = True
                #update the params with the fitted parameters
                if not new_P.shape: #convert from 0D to 1D
                    new_P = numpy.array([new_P])
                self.P     = new_P
                if covar is None:  #on singular matrix
                    covar = Undefined
                self.covar = covar 
            else:
                self.success  = False
        else:
            self.message  = "'P0' was empty, cannot fit"
            self.success  = False

###############################################################################
# TEST CODE
###############################################################################
if __name__ == "__main__":
    from pylab import *
    from numpy import array, exp, linspace
    from numpy.random import normal

    def func(x,A,B):
        return A*x**2 + exp(-B*x)
    
    X = linspace(0,1.0,100)
    noise = normal(loc=0.0,scale=0.1)
    Y = func(X,A=2.4,B=7.6) + noise

    plot(X,Y,'b.')

    def error_func(p,_X,_Y):
        A = p[0]
        B = p[1]
        return _Y - func(_X,A,B)    

    optimizer = NLSOptimizer(error_func, [0.0, 0.0], args=(X,Y))
    optimizer.optimize()
    optimizer.configure_traits()
    P = optimizer.P
    A_fit = P[0]
    B_fit = P[1]
    plot(X, func(X,A=A_fit,B=B_fit),'r-')
    print A_fit,B_fit    
    show()
