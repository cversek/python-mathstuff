###############################################################################
# parameters.py
# desc: a class for keeping track of parameter sets and ordering
# authors: Craig Versek (cversek@physics.umass.edu)
#          Mike Thorn
###############################################################################
from enthought.traits.api import HasTraits, on_trait_change, Float, Bool, Instance, Callable

from enthought.traits.ui.api import ModelView, View, Label, Item, VGrid

from mathstuff.functions.parameters import ParameterSet
###############################################################################

import new

    
class ParameterSetGUI(ModelView):
    model        = Instance(ParameterSet)
    fields       = Instance(HasTraits)
    fields_class = Callable

    view   = View(Item('fields',style='custom',show_label=False))

    @on_trait_change('model')
    def render_fields(self):
        #construct traits for each parameter and place in the class dictionary
        params = self.model.get_params()
        class_dict = {}
        #construct default view
        elements = []
        elements.extend([Label("Parameter"), Label("Error"), Label("Fixed?")])
        for param in params:
            pname = param.name
            class_dict["entry_" + pname] = Float(param.value)  #get the initial value
            class_dict["err_"   + pname] = Float(param.error)  #get the initial value
            class_dict["fixed_" + pname] = Bool(param.fixed)   #determine if fixed or not
            elements.append(Item("entry_" + pname, format_str = "%0.4g",     show_label = True,  label = pname     ))
            elements.append(Item("err_"   + pname, format_str = "+/- %0.2g", show_label = False, style = 'readonly'))
            elements.append(Item("fixed_" + pname, show_label = False                                              ))

        kwargs = {'columns': 3}           
        class_dict['traits_view'] = View(VGrid(*elements, **kwargs))
        self.fields_class = new.classobj('Fields', (HasTraits,), class_dict)
        self.fields = self.fields_class()

    @on_trait_change('fields:entry_+')
    def update_param_value(self, trait_name, new):
        pname = trait_name[6:]                                       #remove the 'entry_' tag
        self.model.update_param(pname, value = new, error = 0.0)
        self.fields.__setattr__("err_" + pname, 0.0)
        #print "update_param_value", trait_name, new

    @on_trait_change('fields:fixed_+')
    def update_param_fixed(self, trait_name, new):
        pname = trait_name[6:]  #remove the 'fixed_' tag
        if new: #set the parameter as fixed
            self.model.update_param(pname, fixed = True, error = 0.0)
            self.fields.__setattr__("err_" + pname, 0.0)
        else:
            self.model.update_param(pname, fixed = False)
       #print "update_param_fixed", trait_name, new


###############################################################################
# TEST CODE
###############################################################################
if __name__ == "__main__":
    p = ParameterSet.from_names(['d','A','B','C'])
    mv = ParameterSetGUI(p)
    mv.configure_traits()
