import numpy as np
from dezero import Variable


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x0, x1):
    y = (1 + (x0 + x1 + 1)**2 * (19 - 14*x0 + 3*x0**2 - 14*x1 + 6*x0*x1 + 3*x1**2)) *\
        (30 + (2*x0 - 3*x1)**2 * (18 - 32*x0 + 12*x0**2 + 48*x1 - 36*x0*x1 + 27*x1**2))
    return y


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
f = goldstein  # sphere / matyas
z = f(x, y)
z.backward()
print(x.grad, y.grad)