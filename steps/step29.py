import os, sys;
if '__file__' in globals():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import dezero
if not dezero.is_simple_core:
    raise RuntimeError('Modify dezero/__init__.py: is_simple_core = True')
from dezero import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)