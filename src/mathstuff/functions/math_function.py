###############################################################################
# math_function.py
# desc:    introspects a python function object to create a mathematical 
#          representation
# authors: Craig Versek (cversek@physics.umass.edu)
#          Mike Thorn
###############################################################################
try:
    #look for the now standard traits.api
    from traits.api import HasTraits, HasStrictTraits, on_trait_change, Trait,\
                                      Instance, Str, Int, Float, Bool, Tuple,\
                                      List, Dict, Set, Undefined, Array, Function
    from traitsui.api import View, Group, Item
except ImportError as err:
    print("Warning caught error: %s" % err)
    #otherwise try to get it from the old enthought tool suite
    from enthought.traits.api import HasTraits, HasStrictTraits, on_trait_change,\
                                     Trait, Instance, Str, Int, Float, Bool,\
                                     Tuple, List, Dict, Set, Undefined, Array,\
                                     Function
    from enthought.traits.ui.api import View, Group, Item

import inspect
import yaml
import numpy, scipy
from   numpy import array, linspace, logspace, hstack
###############################################################################
class MathFunction(HasStrictTraits):
    """ A traits based class for describing and representing a mathematical 
        function from a python function object and additional metadata, encoded
        in the doc_string.

        func: a python function object 
              example: 
                      def func(x,y,A,b,C):
                          "{n_x: 2, n_y: 1}"
                          return A*x + b*y + C
                                          
              defines a 2D -> 1D function (x,y) with 3 parameters A,b,C
                
                       
    """
    #private traits
    _func          = Function 
    _func_repr     = Str

    #public (editable) traits        
    func_name      = Str(repr_ = True)         #the name of the function in the representation
    n_x            = Int(repr_ = True)         #the number of independent variables
    n_y            = Int(repr_ = True)         #the number of dependent variables
    n_p            = Int(repr_ = True)         #the number of the parameters
    n_a            = Int(repr_ = True)         #the length of the argument list to 'func' = n_x + n_p
    
    a_names        = List(Str, repr_ = True)   #all argument names 
    x_names        = List(Str, repr_ = True)   #independent variable names
    p_names        = List(Str, repr_ = True)   #parameter names
    y_names        = List(Str, repr_ = True)   #dependent variable names
    
    formula        = Str(repr_ = True)         #the mathetical formula describing total the function evaluation
    y_formulas     = List(Str,repr_ = True)    #the mathetical formulas describing the evaluation of each depedent variable
    #range_spacing  = Str         #describes the type of range spacing to use to interpolation, 'linear' or 'logarithmic'

    view = View(
                Item('_func_repr'  , label = 'Function Representation',),
                Item('a_names'     , label = 'Arguments'), 
                Item('x_names'     , label = 'Indepedent Variables'),
                Item('p_names'     , label = 'Parameters'),
                Item('y_names'     , label = 'Depedent Variables'),
                resizable = True,
                height = 0.75,
                width  = 0.25
               )    

    def __init__(self, func, **traits):
        self._func  = func
        #discover the traits of the 'func' object by introspection
        init_traits = self._introspect_function_traits()
        #apply any overrides
        init_traits.update(traits)
        #call the super constructor to initialize all the traits
        super(MathFunction, self).__init__(**init_traits)       
        self._create_and_cache_func_repr()

    def __call__(self, *args, **kwargs):
        #make this object callable, like the original function
        return self._func(*args, **kwargs)        
            
    def __repr__(self):    
        return self._func_repr
        
    #-------------------------------------------------------------------------
    def _introspect_function_traits(self): 
        #peer into the functions internals
        func_name = self._func.__name__ 
        func_info = inspect.getdoc(self._func)
        try:
            func_info = yaml.load(func_info)
        except TraitError:
            print "Warning: could not parse 'func_desc' as YAML:", func_info    
        fargs, fvarargs, fvarkwargs, fdefaults = inspect.getargspec(self._func)
        #restrict the allowed function definitions
        if not fvarargs is None:
          raise ValueError, "'func' cannot use varargs"
        elif not fvarkwargs is None:
          raise ValueError, "'func' cannot use varkwargs"
        elif not fdefaults is None:
          raise ValueError, "'func' cannot use defaults"
        n_a = func_info.get('n_a', len(fargs))
        assert n_a == len(fargs)
        n_x         = func_info.get('n_x', 1)
        n_y         = func_info.get('n_y', 1)
        n_p         = n_a - n_x 
          
        a_names     = fargs                     
        x_names     = fargs[   :n_x]  
        p_names     = fargs[n_x:   ]  
        y_names     = func_info.get('y_names', Undefined)  #dependent variable names
        if y_names is Undefined: #create default dep. variable names
            if n_y == 1:
                y_names = ['y'] 
            else:
                y_names = ['y%d' % i for i in xrange(1,n_y + 1)]

        formula        = func_info.get('formula', Undefined)      
        y_formulas     = func_info.get('y_formulas', Undefined)
        #pack into dictionary
        traits = {
                  'func_name': func_name,
                  'n_a': n_a,
                  'n_x': n_x,
                  'n_y': n_y,
                  'n_p': n_p,
                  'a_names': a_names,
                  'x_names': x_names,
                  'y_names': y_names,
                  'p_names': p_names,
                  'formula': formula,
                  'y_formulas': y_formulas,
                 }
        return traits    
    
    @on_trait_change('+repr_') #signifies a representational trait change
    def _create_and_cache_func_repr(self):
        "create mathematical representation of function"
        if self.traits_inited():  #ensure that initialization has completed
            func_repr    = []
            func_repr.append(self.func_name)
            x_repr = ','.join(self.x_names)
            func_repr.append("(%s" % x_repr)
            if self.n_p > 0:
                p_repr       = ','.join(self.p_names) 
                func_repr.append("; %s)" % p_repr)
            else:
                func_repr.append(")")
            if self.n_y == 1:
                func_repr.append(' -> %s' % self.y_names[0])
            else:
                y_formulas = self.y_formulas
                if y_formulas is Undefined:
                    y_formulas = []
                y_list_repr = []
                pairs = map(None,self.y_names,y_formulas)
                for y_name, formula in pairs:
                    if y_name is None:     #too many formulas supplied, ignore the rest
                        break
                    elif formula is None:  #no formula specified
                        y_list_repr.append(y_name)
                    else:
                        y_list_repr.append("%s = %s" % (y_name, formula)) 
                y_list_repr = ",".join(y_list_repr) 
                func_repr.append(' -> <%s>' % y_list_repr)
            if not self.formula is Undefined:
                func_repr.append(" = %s" % self.formula)
            self._func_repr = "".join(func_repr)
        
        
###############################################################################
# TEST CODE
###############################################################################
if __name__ == "__main__":
    from pylab import *
    from numpy.random import normal
    
    def polar_to_cart(r,theta):
        """{y_names: [x,y], y_formulas: ["r*sin(theta)","r*cos(theta)"], n_x: 2, n_y: 2}"""
        return (r*sin(theta),r*cos(theta))

    #MF1 = MathFunction(polar_to_cart)
    #MF1.configure_traits()

    def arrhenius(T,A,E):
        """{n_x: 1, n_y: 1, formula: "A*exp(-E/T)", y_names: ['c']}"""
        return A*exp(-E/T)

    MF2 = MathFunction(arrhenius)
    MF2.configure_traits()

