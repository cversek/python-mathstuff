###############################################################################
# function_fit.py
# desc: convenience interfaces for non-linear curve fitting with the ability
#       to introspect a fit model from a python function object
# authors: Craig Versek (cversek@physics.umass.edu)
#          Mike Thorn
###############################################################################
try:
    #look for the now standard traits.api
    from traits.api import HasTraits, on_trait_change, Trait, Instance, \
                                     Str, Int, Float, Bool, Tuple, List, Dict,    \
                                     Set, Undefined, Array, Function
    from traits.ui.api import View, Group, Item
except ImportError:
    #otherwise try to get it from the old enthought tool suite
    from enthought.traits.api import HasTraits, on_trait_change, Trait, Instance, \
                                     Str, Int, Float, Bool, Tuple, List, Dict,    \
                                     Set, Undefined, Array, Function
    from enthought.traits.ui.api import View, Group, Item

import inspect
import numpy, scipy
from numpy import array, hstack, linspace, logspace, log10, sqrt, ones_like

from mathstuff.functions.math_function import MathFunction
from mathstuff.functions.parameters    import ParameterSet
from mathstuff.algorithms.optimizer    import NLSOptimizer
###############################################################################
class FunctionFit(HasTraits):
    """ A traits based class for simplifying multidimensional nonlinear 
        least squares function fitting.                    
    """
    func           = Instance(MathFunction)
    X              = Array(dtype = numpy.float64,shape=(None,None))  #2D array where each ith row is the data for the ith independent variable in 'func'
    Y              = Array(dtype = numpy.float64,shape=(None,None))  #2D array where each jth row is the data for the ith dependent variable, the value of'func'
    W              = Array(dtype = numpy.float64,shape=(None,None))  #2D array where each jth row is the data for the weighting of the ith dependent variable
    params         = Instance(ParameterSet)                          #hold data structure and methods for ordering and fixed/free parameter designation
    selection_low_index      = Int
    selection_high_index     = Int

    error_func     = Function                                        #automatically generated error function
    optimizer      = Instance(NLSOptimizer)                          #optimizer for error function, obtains best fit parameters
    range_func     = Function                                        #function used for interpolating range
    iter_num       = Int(0)                                          #keep track of the optimizer iterations
    fit_log        = Str("")                                         #stores the info from fitting in a YAML format

    view = View(
                Item('func',      label = 'MathFunction Object',),
                Item('params',    label = 'Parameters'),
                Item('optimizer', label = 'Optimizer',style = 'custom'),
                resizable = True,
                height = 0.75,
                width  = 0.25
               )    

    def __init__(self, func):
        super(FunctionFit, self).__init__()
        self.func       = MathFunction(func)
        
    @on_trait_change('func')
    def update_params(self):
        print "update_params on 'func'"
        p_names = self.func.p_names #extract parameter names
        self.params = ParameterSet.from_names(p_names)
        self.update_error_func()

    def load_data(self, X, Y, W = None):
        #load the data for the independent variables
        X = numpy.array(X)
        d = len(X.shape)
        if d == 1: #promote 1D to 2D
            X = X.reshape((1,-1))
        elif d > 2:
            raise ValueError, "'X' dimension must be 1 or 2, detected incommensurate data of dimension %d" % d
        self.X = X
        #load the data for the dependent variables
        Y = numpy.array(Y)
        d   = len(Y.shape)
        if d == 1: #promote 1D to 2D
            Y = Y.reshape((1,-1))
        elif d > 2:
            raise ValueError, "'Y' dimension must be 1 or 2, detected incommensurate data of dimension %d" % d
        self.Y = Y
        #configure the wieghting data
        if W is None:
            W = ones_like(self.Y)  #use unity weighting by default
        W = numpy.array(W)
        assert W.shape == Y.shape
        self.W = W
        #initialize the indices for data fitting
        self.selection_low_index  = 0
        self.selection_high_index = self.X.shape[1]        

    def get_data(self):
        X = self.X[:]
        Y = self.Y[:]
        W = self.W[:]
        return (X, Y, W)   

    def get_selection(self, selection_low_index =  None, selection_high_index = None):
        if selection_low_index is None:
            selection_low_index = self.selection_low_index
        if selection_high_index is None:
            selection_high_index = self.selection_high_index
        if selection_low_index > selection_high_index:
            selection_low_index, selection_high_index = selection_high_index, selection_low_index
        X = self.X[:,selection_low_index: selection_high_index]
        Y = self.Y[:,selection_low_index: selection_high_index]
        W = self.W[:,selection_low_index: selection_high_index]
        return (X, Y, W)         
    
    def fit(self):
        self.update_error_func() 
        error_func = self.error_func

        info = {} #for storing fitting informtion

        P0 = self.params.get_values(free = True)
        info['P0'] = P0
        
        self._clear_log() #empty the fitting log
        self._print_log("## Starting Fit ##") 
        if len(P0) > 0:  #free parameter set cannot be empty for fitting
            self.iter_num = 0 #reset iteration counter
            self.optimizer = NLSOptimizer(cost_map = error_func, P0 = P0)
            self.optimizer.optimize()
            #determine if fitting was successful
            success  = self.optimizer.success
            msg      = self.optimizer.message
            info['success'] = success
            info['msg']     = msg
            if success:
                #update the parameters
                fp_values  = self.optimizer.P
                fp_names = self.params.get_names(free = True)
                for name, value in zip(fp_names,fp_values):
                    self.params.update_param(name,value = value)
                #compute the error on the parameters, if possible
                err   = self.optimizer.cost
                ndf   = self.optimizer.ndf
                reduced_chisqr = (err*err).sum()/ndf 
                covar = self.optimizer.covar
                if not covar is Undefined:  
                    covar *= reduced_chisqr #rescale the covariance matrix
                    p_var = covar.diagonal()
                    p_err = sqrt(p_var)
                    for name, error in zip(fp_names,p_err):
                        self.params.update_param(name,error = error)
                self._print_log("## Fitting Completed ##")
                self._print_log("---")
                self._print_log("parameters:")
                self._print_log(self.params, level = 2, indent = '  ')
                self._print_log("ndf: %d" % ndf)
                self._print_log("reduced_chisqr: %g" % reduced_chisqr)
                
                info['err'] = err
                info['ndf'] = ndf
                info['reduced_chisqr'] = reduced_chisqr
                info['covar'] = covar
                info['param_values'] = self.params.get_value_dict()
                info['param_errors'] = self.params.get_error_dict()
            else:
                self._print_log("## Fitting Failed! ##")
                self._print_log("---")
            self._print_log("ierr: %s" % self.optimizer.ier)
            self._print_log("message: %s" % msg)
            self._print_log("...")
        else:   #empty free parameter set, do not fit 
            pass
        return info

    def evaluate(self, X = None, pdict = None):
        """evaluatue the model function with supplied arguments and the current parameter set, 
           or by default, from the stored data sets"""
        if X is None:                  #evaluate the function with the current X data
            X = self.X
        if pdict is None:
            pdict = self.params.get_value_dict() #evaluate the function with the current parameter set
        Y = self.func(*X,**pdict)
        Y = numpy.array(Y)
        d = len(Y.shape)
        if d == 1: #promote 1D to 2D
            Y = Y.reshape((1,-1))
        elif d > 2:
            raise ValueError, "'Y' dimension must be 1 or 2, detected incommensurate data of dimension %d" % d
        return Y
        

    def interpolate(self, X_ranges = None, density = 1, range_func = None, range_spacing = 'linear'):
        """evaluatue the model function:
               by default, from the min and max of the supplied data at a 'density'
               or between the supplied 'X_ranges' = [(x1_min,x1_max,[num_x1]), ... ]
        """
        #set up the function to evaluate the ranges
        if range_spacing == 'linear':
            range_func = linspace
        elif range_spacing == 'logarithmic':
            range_func = logspace
        else:
            raise ValueError, "unknown 'range_spacing' type: %s" % range_spacing
    
        if X_ranges is None:
            X_ranges = [None]*self.X.shape[0]

        X = []
        for x, rang in map(None, self.X, X_ranges):  #pair elements of X with ranges, filling in with Nones
            if x is None:
                raise ValueError, "len(X_ranges) cannot be > self.X.shape[0]"
            if rang is None:
                low  = x.min()
                high = x.max()
                n    = len(x)
                if range_spacing == "logarithmic":
                    low  = log10(low)
                    high = log10(high)         
                X.append( range_func(low,high,n*density) )
            elif len(rang) == 2:  #no num points specified, use the density parameter
                X.append( range_func(r[0],r[1],len(x)*density) )
            elif len(rang) == 3:  #full range specified
                X.append( range_func(r[0],r[1],r[3]) )
            else:
                raise ValueError, "each element of 'X_ranges' must be None, (x_min,x_max), or (x_min,x_max, num_x)"
                 
        X = array(X)
        Y = self.evaluate(X)
        return (X, Y)

    def update_error_func(self):
        #freeze out copies of data
        fp_names = self.params.get_names(free = True)
        pdict    = self.params.get_value_dict()
        X, Y, W  = self.get_selection()
        #create the function closure on the free (varied) parameter set
        func = self.evaluate
        def varied_func(p):
            pdict.update( dict( [(fp_name,val) for fp_name, val in zip(fp_names,p)] ) )
            return func(X = X, pdict = pdict)      
        #create the error function closure:
        def error_func(p):
            F = varied_func(p)                          #evaulate the varied function on the parameter set, p
            errs = [(y - f)*w for y,f,w in zip(Y,F,W)]  #pair each data row with its function evalutation element and weighting
            errs = hstack(errs)                         #create a lumped row vector of all the deviations
            #amplify error for negative parametes
            for p_i in p:
                if p_i < 0:
                    errs *= (1.0 + abs(p_i))
            return errs 
        self.error_func = error_func

    def _clear_log(self):
        self.fit_log = ""
        
    def _print_log(self, text, indent = "\t", level = 0, newline = '\n'):
        text = str(text)
        if level >= 1: #reformat the text to indent it
            text_lines = text.split(newline)
            space = indent*level
            text_lines = ["%s%s" % (space, line) for line in text_lines]
            text = newline.join(text_lines)
        self.fit_log += text + newline
        
        
###############################################################################
# TEST CODE
###############################################################################
if __name__ == "__main__":
    from pylab import *
    from numpy.random import normal
    
    def curve(x,A,B):
        """
           n_x: 1 
           n_y: 1
           formula: 'A*x + B'
        """
        return A*x + B

    X = linspace(-10, 10, 100)
    noise = normal(loc=0.0,scale=1.0,size=len(X))
    Y = curve(X,A=1.0, B=2.0) + noise

    fitter = FunctionFit(curve)
    fitter.load_data(X,Y)
    fitter.fit()
    fitter.configure_traits()
    plot(X,Y,'b.',X,curve(X,A=fitter.params['A'],B=fitter.params['B']),'r-')
    print fitter.params.params   
    show()

#### EOF ###########################################################################
