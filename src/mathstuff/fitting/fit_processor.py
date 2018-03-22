###############################################################################
# function_fit.py
# desc: convenience interfaces for non-linear curve fitting with the ability
#       to introspect a fit model from a python function object
# authors: Craig Versek (cversek@physics.umass.edu)
#          Mike Thorn
###############################################################################
from enthought.traits.api import HasTraits, on_trait_change, Trait, Instance, \
                                 Str, Int, Float, Bool, Tuple, List, Dict,    \
                                 Set, Undefined, Array, Function
from enthought.traits.ui.api import View, Group, Item
import inspect
import numpy, scipy
from numpy import array, hstack, linspace, logspace, log10, sqrt, ones_like

from mathstuff.fitting.fit_model      import FitModel
from mathstuff.fitting.fit_data       import FitData
from mathstuff.algorithms.optimizer   import NLSOptimizer
###############################################################################
class FitProcessor(HasTraits):
    """ A traits based class for simplifying multidimensional nonlinear 
        least squares function fitting.                    
    """
    fit_data       = Instance(FitData)        #holds the data
    fit_model      = Instance(FitModel)       #holds the data selection, fit parameters, and functional model     

    error_func     = Function                 #automatically generated error function
    optimizer      = Instance(NLSOptimizer)   #optimizer for error function, obtains best fit parameters
    iter_num       = Int(0)                   #keep track of the optimizer iterations
    fit_log        = Str("")                  #stores the info from fitting in a YAML format

    view = View(
                Item('optimizer', label = 'Optimizer',style = 'custom'),
                resizable = True,
                height = 0.75,
                width  = 0.25
               )
    #--------------------------------------------------------------------------
    #@on_trait_change('fit_model')    
    def update_error_func(self):
        #freeze out copies of data and parameters
        fp_names = self.fit_model.get_free_param_names()
        pdict    = self.fit_model.get_params_dict()
        X, Y, W  = self.fit_data.get_selection()
        #create the function closure on the free (varied) parameter set
        func = self.fit_model.evaluate
        def varied_func(p):
            pdict.update( dict( [(fp_name,val) for fp_name, val in zip(fp_names,p)] ) )
            return func(X = X, pdict = pdict)      
        #create the error function closure:
        def error_func(p):
            F = varied_func(p)                          #evaulate the varied function on the parameter set, p
            errs = [(y - f)*w for y,f,w in zip(Y,F,W)]  #pair each data row with its function evalutation element and weighting
            errs = hstack(errs)                         #create a lumped row vector of all the deviations
            return errs 
        self.error_func = error_func

    #--------------------------------------------------------------------------
    def fit(self):
        "run the optimizer on the error function to obtain best fit parameters"""
        error_func = self.error_func
        P0 = self.fit_model.get_free_param_values()
        self._clear_log() #empty the fitting log
        self._print_log("## Starting Fit ##") 
        if len(P0) > 0:  #free parameter set cannot be empty for fitting
            self.iter_num = 0 #reset iteration counter
            self.optimizer = NLSOptimizer(cost_map = error_func, P0 = P0)
            self.optimizer.optimize()
            #determine if fitting was successful
            success  = self.optimizer.success
            msg      = self.optimizer.message
            if success:
                #update the parameters
                fp_values  = self.optimizer.P
                fp_names   = self.fit_model.get_free_param_names()
                for name, value in zip(fp_names,fp_values):
                    self.fit_model.update_param(name,value = value)
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
                        self.fit_model.update_param(name,error = error)
                self._print_log("## Fitting Completed ##")
                self._print_log("---")
                self._print_log("parameters:")
                self._print_log(self.fit_model.params, level = 2, indent = '  ')
                self._print_log("ndf: %d" % ndf)
                self._print_log("reduced_chisqr: %g" % reduced_chisqr)
            else:
                self._print_log("## Fitting Failed! ##")
                self._print_log("---")
            self._print_log("ierr: %s" % self.optimizer.ier)
            self._print_log("message: %s" % msg)
            self._print_log("...")
        else:   #empty free parameter set, do not fit 
            pass

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
# TEST CODE - FIXME
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
    fitter.fit(X, Y)
    fitter.configure_traits()
    plot(X,Y,'b.',X,curve(X,A=fitter.params['A'],B=fitter.params['B']),'r-')
    print fitter.params.params   
    show()

#### EOF ###########################################################################
