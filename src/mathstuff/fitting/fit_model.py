###############################################################################
# fit_model.py
# desc:    combines a math_function model for data with other tools for
#          describing robust fitting procedures
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

from mathstuff.functions.math_function import MathFunction
from mathstuff.functions.parameters    import ParameterSet
###############################################################################
class FitModel(HasTraits):
    """ A traits based class for simplifying multidimensional nonlinear 
        least squares function fitting.                    
    """
    func           = Instance(MathFunction)   #mathematical model to be fit
    params         = Instance(ParameterSet)   #hold data structure and methods for ordering and fixed/free parameter designation
    selection_low_index      = Int
    selection_high_index     = Int

    view = View(
                Item('func',      label = 'MathFunction Object',),
                Item('params',    label = 'Parameters'),
                resizable = True,
                height = 0.75,
                width  = 0.25
               )    
    #--------------------------------------------------------------------------
    def __init__(self, func):
        super(FunctionFit, self).__init__()
        self.func       = MathFunction(func)
    #--------------------------------------------------------------------------    
    @on_trait_change('func')
    def update_params(self):
        p_names = self.func.p_names #extract parameter names
        self.params = ParameterSet.from_names(p_names)
    #--------------------------------------------------------------------------
    def get_params_dict(self)
        return self.params.get_value_dict()    

    def get_free_param_names(self)
        return self.params.get_names(free = True)

    def get_free_param_values(self)
        return self.params.get_values(free = True)

    def update_param(self, name, value = None, error = None)
        self.params.update_param(name,value = value, error = error)

    def evaluate(self, X, pdict = None):
        """evaluatue the model function with supplied arguments and the current parameter set, 
           or by default, from the stored data sets"""
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
        
###############################################################################
# TEST CODE - FIXME
###############################################################################
if __name__ == "__main__":
    from pylab import *
    from numpy.random import normal
    
#    def polar_to_cart(r,theta):
#        """{y_names: [x,y], y_formulas: ["r*sin(theta)","r*cos(theta)"], n_x: 2, n_y: 2}"""
#        return (r*sin(theta),r*cos(theta))

#    #MF1 = MathFunction(polar_to_cart)
#    #MF1.configure_traits()

#    def arrhenius(T,A,E):
#        """{n_x: 1, n_y: 1, formula: "A*exp(-E/T)", y_names: ['c']}"""
#        return A*exp(-E/T)

#    FM = FitModel(arrhenius)
#    #FM.func_name = "POOP"
#    FM.configure_traits()

