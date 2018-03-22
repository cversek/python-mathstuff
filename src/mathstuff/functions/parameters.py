###############################################################################
# parameters.py
# desc: a class for keeping track of parameter sets and ordering
# authors: Craig Versek (cversek@physics.umass.edu)
#          Mike Thorn
###############################################################################
from enthought.traits.api import HasTraits, HasStrictTraits, on_trait_change, Tuple, Dict, Set,\
                                 Undefined, Str, Float, List, Bool, Instance, Class, Callable

##############################################################################
class Parameter(HasStrictTraits):
    name  = Str
    value = Float(0.0)
    error = Float(0.0)
    fixed = Bool(False)

##############################################################################
class ParameterSet(HasTraits):
    """ A traits enabled class to maintain sets of free and fixed Parameter 
        objects and to supply a standard ordering.
    """
    pkey_list   = List(Str)
    pdict       = Dict(Str,Parameter)

    #--------------------------------------------------------------------------
    def __init__(self, params):
        super(ParameterSet,self).__init__()
        #get the parameters to use as indices, filtering out repeats
        pkeys = []
        pitems = []
        for param in params:
            pname = param.name
            if not pname in pkeys:
                pkeys.append(pname)
                pitems.append((pname,param))
        self.pkey_list = pkeys  
        self.pdict     = dict(pitems)
    #--------------------------------------------------------------------------
    @classmethod
    def from_names(cls, 
                   names, 
                   init_values  = None, 
                   init_default = 0.0, 
                   ):
        #set up the params dict
        params = []
        if init_values is None:
            #make all values init_default
            init_values = [None]*len(names)
        #replace None values with init_default
        for name, value in map(None,names,init_values):
            if not name is None:
                if value is None:
                    value = init_default
                params.append(Parameter(name = name, value = value))
        return cls(params)
    #--------------------------------------------------------------------------
    # trait update handlers
    @on_trait_change('pdict:+')
    def _update_pname(self, obj, trait_name, old, new):
        if trait_name == 'name':  
            #change the key in the pdict and plist
            del self.pdict[old]
            self.pdict[new] = obj
            i = self.pkey_list.index(old)
            self.pkey_list[i] = new
        
    #--------------------------------------------------------------------------
    def __str__(self):
        buff = []
        for pname in self.pkey_list:
            param = self.pdict[pname]
            line = "%s: %g +/- %g" % (param.name,param.value,param.error)
            if param.fixed:
                line += " (fixed)"
            buff.append(line)
        buff = '\n'.join(buff)
        return buff

    #--------------------------------------------------------------------------
    # accessor methods
    def get_param(self, key):
        return self.pdict[key]
  
    def get_params(self):
        return [self.pdict[key] for key in self.pkey_list]

    def get_names(self, free = None):
        if free is None:
            return [param.name for param in self.get_params()]
        else:
            return [param.name for param in self.get_params() if not param.fixed == free]

    def get_values(self, free = None):
        if free is None:
            return [param.value for param in self.get_params()]
        else:
            return [param.value for param in self.get_params() if not param.fixed == free]

    def get_value_dict(self, free = None):
        if free is None:
            return dict([(param.name,param.value) for param in self.get_params()])
        else:
            return dict([(param.name,param.value) for param in self.get_params() if not param.fixed == free])

    def get_error_dict(self, free = None):
        if free is None:
            return dict([(param.name,param.error) for param in self.get_params()])
        else:
            return dict([(param.name,param.error) for param in self.get_params() if not param.fixed == free])

   #--------------------------------------------------------------------------
    # mutator methods
    def add_param(self, param):
        name = param.name
        self.pdict[name] = param         

    def add_params(self, params):
        for param in params:
            self.add_param(param)
            
    def update_param(self, key, **kwargs):
        if not key in self.pdict.keys():
            raise KeyError, "'%s' is not a valid parameter key" % key 
        param = self.get_param(key)
        param.trait_set(**kwargs)  

###############################################################################
# TEST CODE
###############################################################################
if __name__ == "__main__":
    PS = ParameterSet.from_names(['d','A','B','C'])
    PS.configure_traits()
