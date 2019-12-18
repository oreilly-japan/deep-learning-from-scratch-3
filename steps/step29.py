import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
# Import core_simple explicitly
from dezero.core_simple import Variable
from dezero.core_simple import setup_variable
setup_variable()


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


logs = []
x = Variable(np.array(2.0))
iters = 200

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= 0.01 * x.grad