import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
# Import core_simple explicitly
from dezero.core_simple import Variable
from dezero.core_simple import setup_variable
setup_variable()


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
iters = 1000
lr = 0.001

for i in range(iters):
    print(x0, x1)

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad