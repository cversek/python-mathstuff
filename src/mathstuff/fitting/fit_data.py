###############################################################################
# fit_data.py
# desc:    
# authors: Craig Versek (cversek@physics.umass.edu)
#          Mike Thorn
###############################################################################
from enthought.traits.api import HasTraits, on_trait_change, Trait, Instance, \
                                 Str, Int, Float, Bool, Tuple, List, Dict,    \
                                 Set, Undefined, Array, Function
from enthought.traits.ui.api import View, Group, Item
from numpy import array, hstack, linspace, logspace, log10, sqrt, ones_like
###############################################################################
class FitData(HasTraits):
    """                   
    """
    X = Array(dtype = numpy.float64,shape=(None,None))  #2D array where each ith row is the data for the ith independent variable in 'func'
    Y = Array(dtype = numpy.float64,shape=(None,None))  #2D array where each jth row is the data for the ith dependent variable, the value of'func'
    W = Array(dtype = numpy.float64,shape=(None,None))  #2D array where each jth row is the data for the weighting of the ith dependent variable
    #--------------------------------------------------------------------------
    def load_data(self, X, Y, W = None):
        #load the data for the independent variables
        X = array(X)
        d = len(X.shape)
        if d == 1: #promote 1D to 2D
            X = X.reshape((1,-1))
        elif d > 2:
            raise ValueError, "'X' dimension must be 1 or 2, detected incommensurate data of dimension %d" % d
        self.X = X
        #load the data for the dependent variables
        Y = array(Y)
        d   = len(Y.shape)
        if d == 1: #promote 1D to 2D
            Y = Y.reshape((1,-1))
        elif d > 2:
            raise ValueError, "'Y' dimension must be 1 or 2, detected incommensurate data of dimension %d" % d
        self.Y = Y
        #configure the wieghting data
        if W is None:
            W = ones_like(self.Y)  #use unity weighting by default
        W = array(W)
        assert W.shape == Y.shape
        self.W = W    

    def get_data(self):
        X = self.X[:]
        Y = self.Y[:]
        W = self.W[:]
        return (X, Y, W)   

    def get_selection(self, low_index, high_index):
        if low_index > high_index: #swap if order reversed
            low_index, high_index = high_index, low_index
        X = self.X[:,low_index: high_index]
        Y = self.Y[:,low_index: high_index]
        W = self.W[:,low_index: high_index]
        return (X, Y, W)         
        
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

