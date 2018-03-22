import pylab
from numpy import array
from enthought.traits.api import HasTraits, on_trait_change, Trait, Instance, \
                                 Str, Int, Float, Bool, Tuple, List, Dict,    \
                                 Set, Undefined, Array, Function, Interface,  \
                                 implements, Button

from enthought.traits.ui.api import View, Item, Group, ModelView, VGroup, HSplit

from mathstuff.plotting.wx_mpl_figure_editor import Figure, MPLFigureEditor

###############################################################################
class IPlot(Interface):
    """ An interface defining an object which can render a plot on a 
        figure object
    """
    def clear(self):
        """ """
    def render(self, Xs, Ys, figure = None, **kwargs):
        """ Plots data from 'Xs', 'Ys' on 'figure'  and returns the 
            figure object """

    def redraw(self):
        """ """

###############################################################################
class BasePlot(HasTraits):
    """ An interface defining an object which can render a plot on a 
        figure object
    """
    implements(IPlot)
    n_x    = Int(1)
    n_y    = Int(1)
    figure = Instance(Figure,())
    view   = View(
                  Item('figure', 
                       #height = 600,
                       #width  = 800,
                       style  = 'custom',
                       show_label = False,
                       editor = MPLFigureEditor(), #this editor will automatically find and connect the _handle_onpick method for handling matplotlib's object picking events )
                      ),
                 )             

    def clear(self):
        self.figure.clear()
    
    def render(self, Xs, Ys, fmts = None, labels = None, pickable = [], **kwargs):
        ''' Plots data from 'Xs', 'Ys' on 'figure'  and returns the 
            figure object'''
        data = self._convert_data(Xs,Ys) 
        Xs = data['Xs']
        Ys = data['Ys']
        if fmts is None:
            fmts   = []
        if labels is None:
            labels = [] 
        axes = self.figure.add_subplot(111)
        #kwargs['axes']   = axes
        #kwargs['figure'] = self.figure
        for X,Y,fmt,label in map(None, Xs, Ys, fmts, labels):
            if not (X is None or Y is None):
                kwargs['label'] = label
                self._plot(X,Y,fmt,axes = axes, **kwargs)      
        if labels:
            axes.legend()
        #set up the plot point object selection
        for ind in pickable:
            line = axes.lines[ind]
            line.set_picker(5.0)
    
    def redraw(self):
        if not self.figure.canvas is None:
            self.figure.canvas.draw()

    def register_onpick_handler(self, handler):
        self._handle_onpick = handler 

    def _plot(self, x, y, fmt = None, axes = None, **kwargs):
        if axes is None:
            raise TypeError, "an 'axes' object must be supplied"               
        if fmt is None:
            axes.plot(x,y,**kwargs)
        else:
            axes.plot(x,y,fmt,**kwargs)      

    def _convert_data(self, Xs = None, Ys = None):
        #convert the data for the independent variables
        data_args = {'Xs':(Xs, self.n_x),'Ys':(Ys, self.n_y)}
        data      = {}
        for name, args in data_args.items():
            D, n = args         #data array, expected number of variables
            if not D is None:
                for d in D:
                    print d.shape
                D = array(D)        #convert to a numpy array
                dim = len(D.shape)
                if dim == 1:
                    if n == 1:        
                        #upconvert 1D array to 2D
                        D = D.reshape((1,-1))
                    else:
                        raise TypeError, "'%s' dimension must be 2 or 3 for n > 1, detected incommensurate data of dimension %d" % (name,dim)
                elif dim == 2:
                    d1, d2 = D.shape
                    if n == 1:
                        pass #no conversion needed
                    elif not(d1 == n):
                        raise TypeError, "'%s' shape (%d,%d) must match (n=%d,:)" % (name,d1,d2,n) 
                    else:
                        #up convert 2D array to 3D
                        D = D.reshape((1,d1,d2))
                elif dim == 3:
                    d1, d2, d3 = D.shape
                    if n == 1 and d2 == 1:
                        #down convert 3D array to 2D
                        D = D.reshape((d1,d3))
                    elif not(d2 == n):
                        raise TypeError, "'%s' shape (%d,%d,%d) must match (:,n=%d,:)" % (name,d1,d2,d3,n)
                else:
                    raise TypeError, "'%s' dimension must be 1, 2 or 3, detected incommensurate data of dimension %d" % (name,dim)          
                data[name] = D
            else:
                #default to an empty array
                data[name] = array([])
        return data

###############################################################################
class Function1DCurve(BasePlot):
    """
    """
    n_x = 1  #one independent variable
    n_y = 1  #one dependent variable

###############################################################################
class ParametricCurve(BasePlot):
    """ 
    """
    n_x = -1 #any number of independent variable
    n_y = 2  #two independent variable
    implements(IPlot)
    def render(self, Xs = None, Ys = None, fmts = None, labels = None, axis_scaling = 'equal', pickable = [], **kwargs):
        data = self._convert_data(Ys = Ys) 
        Ys = data['Ys']  
        if fmts is None:
            fmts   = []
        if labels is None:
            labels = []       
        axes = self.figure.add_subplot(111)
        for Y,fmt, label in map(None, Ys, fmts, labels):
            kwargs['label'] = label
            #array Y contains the x,y point data
            if not Y is None:
                x,y = Y
                self._plot(x,y,fmt, axes = axes, **kwargs)
        #post-plot adjustments
        axes.axis(axis_scaling) #set the relative scaling of the axes
        if labels:
            axes.legend()
        #set up the plot point object selection
        for ind in pickable:
            line = axes.lines[ind]
            line.set_picker(5.0)
    
###############################################################################
# TEST CODE - FIXME
###############################################################################        
if __name__ == "__main__":
    from numpy import *
    
    #test FunctionPlot1D
    def curve(x,A,B,C):
        """
           n_x: 1 
           n_y: 1
           formula: 'A*exp(-(x/B)**2) + C'
        """
        return A*exp(-(x/B)**2) + C
   

    X = linspace(-10, 10, 100)
    Y1 = curve(X,A=1.0, B=2.0, C=3.0)
    Y2 = curve(X,A=1.5, B=2.5, C=2.5)

    
    #test data formatting
    plot = Function1DCurve()
    plot.clear()
    plot.render(Xs = X,Ys = Y1, fmts = ['r.','b.'])
    plot.redraw()
    plot.configure_traits()
    plot = Function1DCurve()
    plot.clear()
    plot.render(Xs = [X,X],Ys = [Y1,Y2], fmts = ['r.','b.'], labels = ['f1','f2'])
    plot.redraw()
    plot.configure_traits()
    plot = Function1DCurve()
    plot.clear()
    plot.render(Xs = [[X],[X]],Ys = [[Y1],[Y2]], fmts = ['r.','b.'], labels = ['f1','f2'])
    plot.redraw()
    plot.configure_traits()
    plot.clear()
    
    
    #test ParametricCurve

    def Z_arc(w,R1,C1,R2,C2):
        """
           n_x: 1 
           n_y: 2
           y_names: ['Z_re','Z_im']
           formula: 'R/(1 + j*R*C*w)'
           range_spacing: logarithmic
        """
        Z = R1/(1.0 + 1.0j*R1*C1*w) + R2/(1.0 + 1.0j*R2*C2*w)
        return (Z.real, Z.imag)


    #simulate data
    high_ord = 7
    low_ord  = 2
    ppd      = 10
    w = logspace(high_ord,low_ord,ppd*(high_ord - low_ord))

    Z_re1, Z_im1 = Z_arc(w,R1=1.0e6,C1=1e-12,R2=2.0e6,C2=1e-10)
    Z_re2, Z_im2 = Z_arc(w,R1=1.5e6,C1=1e-10,R2=1.5e6,C2=1e-12)

    #test data formatting
    plot = ParametricCurve()
    plot.clear()
    plot.render(Ys = [(Z_re1,Z_im1),(Z_re2,Z_im2)], fmts = ['r.','b.','g.'], labels = ['Z1',None])
    plot.redraw()
    plot.configure_traits()
    
