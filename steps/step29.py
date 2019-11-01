import numpy as np
from dezero import Variable


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

    x.data -= 0.01 * x.grad.data