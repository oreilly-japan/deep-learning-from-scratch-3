'''
Need the dot binary from the graphviz package (www.graphviz.org).
'''
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
from steps.step24 import goldstein


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = goldstein(x0, x1)
y.backward()

x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'
plot_dot_graph(y, verbose=False)