###############################################################################
import time
import numpy, pylab

from enthought.traits.api import HasTraits, on_trait_change, Trait, Instance, \
                                 Str, Int, Float, Bool, Tuple, List, Dict,    \
                                 Set, Undefined, Array, Function, Interface,  \
                                 implements, Button
      
from enthought.traits.ui.api import View, Item, Group, ModelView, VGroup, HSplit, VGrid
from enthought.traits.ui.editors import TextEditor


from mathstuff.fitting.function_fit import FunctionFit

from mathstuff.functions.parameters_gui import ParameterSetGUI  
from mathstuff.plotting.plots import IPlot  

###############################################################################        
class FitGUI(ModelView):
    """ Implements a Model-View style controller and default view that provides 
        a matplolib graphical representation of an ImpedanceAnalysis model.
    """
    text_display    = Str
    params_gui      = Instance(ParameterSetGUI)
    fit_button      = Button()
    plot            = Instance(IPlot)
    has_fit         = Bool(False)
    pick_mode       = Int(0)
    

    view = View( 
                 HSplit(
                        VGroup( Item('text_display', 
                                     style='custom',
                                     show_label=False,
                                     #springy = True,
                                     height=0.50, 
                                     width=0.20,
                                    ),
                               # VGrid(
                                     # Item('model.selection_low_index',  width=0.05,),
                                     # Item('model.selection_high_index', width=0.05,),
                                      Item('params_gui', 
                                           show_label=False,
                                           style='custom',
                                           springy = True,
                                           #height=0.30, 
                                           width=0.20,
                                          ),
                                      Item('fit_button',
                                           label = "Run Fit",
                                           show_label=False,
                                           #style='custom',
                                           #springy = True,
                                           #height=0.30, 
                                           width=0.20,
                                          ),
                                 #     columns = 1,
                                 #    ),   
                             ),
                        Item('plot', 
                              #editor = MPLFigureEditor(), #this editor will automatically find and connect the _handle_onpick method for handling matplotlib's object picking events
                              style = 'custom',
                              show_label=False,
                              height=0.80, 
                              width=0.60,
                             ),
                       ),
                 height=0.80, 
                 width=0.80,
                 resizable=True,
               )
    #--------------------------------------------------------------------------
    def run_fit(self):
        self.model.fit()
        self.has_fit = True
        new_params = self.model.params
        self.params_gui.model = new_params
        self.params_gui.render_fields()     #FIXME - to use a gentler field update
        self._print(self.model.fit_log, loc = 'front')
        self.update_plot()

#    def _plot_default(self):
#        return SimpleFunctionPlot()

    @on_trait_change('model')
    def update_all(self):
        self.params_gui = ParameterSetGUI(self.model.params)
        self._clear_display()
        self.update_plot()

    @on_trait_change('plot')
    def setup_onpick_handler(self):
        self.plot.register_onpick_handler(self._handle_onpick)
        
    def update_plot(self):
        self.plot.clear()
        Xs = []
        Ys = []
        #get all the data
        X_data, Y_data, W_data = self.model.get_data()
        Xs.append(X_data)
        Ys.append(Y_data)
        #get the data selection
        X_sel, Y_sel, W_sel = self.model.get_selection()
        Xs.append(X_sel)
        Ys.append(Y_sel)
        if self.has_fit:        
            X_fit,  Y_fit  = self.model.interpolate(density = 10)
            Xs.append(X_fit)
            Ys.append(Y_fit)
        self.plot.render(Xs,Ys, fmts = ['b.','kx','r-'], pickable = [0])
        self.plot.redraw()

    def _fit_button_fired(self):
        self.run_fit()
        

    #--------------------------------------------------------------------------
    def _handle_onpick(self,event):
        "handles a data point pick event depending on the mode"
        #extract information from the event
        line = event.artist
        ind = event.ind[0]  #the selected data index as first (nearest) in the group
        X = line.get_xdata()
        Y = line.get_ydata()
        x = X[ind]
        y = Y[ind]
        pick_mode = self.pick_mode
        print event.__dict__
        if pick_mode == 0:
            self._print("\nlow point selected at index: %s (%e, %e)" % (ind,x,y), loc = 'front')
            self.model.selection_low_index  = ind
            self.model.selection_high_index = Undefined
            self.pick_mode = 1
            time.sleep(0.1)
        elif pick_mode == 1:
            self._print("\nhigh point selected at index: %s (%e, %e)" % (ind,x,y), loc = 'front')
            self.model.selection_high_index = ind
            if self.model.selection_low_index != self.model.selection_high_index:
                self.run_fit()
            self.pick_mode = 0
        

    #--------------------------------------------------------------------------
    def _clear_display(self):
        self.text_display = ""

    def _print(self, text, indent = "\t", level = 0, newline = '\n', loc = 'front'):
        text = str(text)
        if level >= 1: #reformat the text to indent it
            text_lines = text.split(newline)
            space = indent*level
            text_lines = ["%s%s" % (space, line) for line in text_lines]
            text = newline.join(text_lines)
        if loc == 'end':
            self.text_display += text + newline
        elif loc == 'front':
            self.text_display = text + newline + self.text_display 
        else:
            raise ValueError, "'loc' must be 'front' or 'end'"
   

###############################################################################
# TEST CODE - FIXME
###############################################################################        
if __name__ == "__main__":
    from function_fit import FunctionFit
    from numpy.random import normal
    from numpy import *
    
#    def curve(x,A,B,C):
#        """
#           n_x: 1 
#           n_y: 1
#           formula: 'A*x + B'
#        """
#        return A*exp(-(x/B)**2) + C
#   

#    X = linspace(-10, 10, 100)
#    noise = normal(loc=0.0,scale=0.1,size=len(X))
#    Y = curve(X,A=1.0, B=2.0, C=3.0) + noise

#    fit = FunctionFit(curve)
#    fit.load_data(X, Y)

    #adapt the ImpedanceThreeViewPlot to the IPlot interface
    
#            #set up the plot point object selection
#            for line in ax.lines:
#                line.set_picker(5.0)
#              

    def Z_arc1(w,R0,R,C):
        """
           n_x: 1 
           n_y: 2
           y_names: ['Z_re','Z_im']
           formula: 'R/(1 + j*R*C*w)'
           range_spacing: logarithmic
        """
        Z = R0 + R/(1.0 + 1.0j*R*C*w)
        return (Z.real, Z.imag)
    

    def Z_arc2(w,R1,C1,R2,C2):
        """
           n_x: 1 
           n_y: 2
           y_names: ['Z_re','Z_im']
           formula: 'R/(1 + j*R*C*w)'
           range_spacing: logarithmic
        """
        Z = R1/(1.0 + 1.0j*R1*C1*w) + R2/(1.0 + 1.0j*R2*C2*w)
        return (Z.real, Z.imag)


    #simulate impedance data
    high_ord = 7
    low_ord  = 2
    ppd      = 10
    w = logspace(high_ord,low_ord,ppd*(high_ord - low_ord))
    Z_re, Z_im = Z_arc2(w,R1=1.0e6,C1=1e-12,R2=2.0e6,C2=1e-10)
#    Z_re, Z_im = ZY_arc(w,R0 = 0.5e6,R1=1.0e6,C1=1e-12,R2=2.0e6,C2=1e-10)

    #add proportional noise to data
    n1 = normal(loc=0.0,scale=0.01,size=len(Z_re))
    n2 = normal(loc=0.0,scale=0.01,size=len(Z_im))
    Z_re += Z_re*n1
    Z_im += Z_im*n2
    #compute weightings for the impedance data
    #W = (1.0/n1**2, 1.0/n2**2)
    W = (1.0/Z_re, 1.0/Z_im)
    #W = (1.0/Z_re, 1.0/Z_im)

    fit = FunctionFit(Z_arc1)
    fit.load_data(X = w, 
                  Y = (Z_re, Z_im), 
                  W = W)


    #pg = ParametersGUI(fit.params)
    #pg.configure_traits()
    #fitter.configure_traits()
    
    #print fit.params
    #fit.fit(X,Y)  

    #print fit.params
    #fit.configure_traits()
    
    #IA.configure_traits()
    
    from pyEIS.core.plotting.eis_plots import ImpedanceThreeViewIPlotAdapter
    fit_gui = FitGUI(model = fit, plot = ImpedanceThreeViewIPlotAdapter())
    #fit_gui.plot = 
    fit_gui.configure_traits()
