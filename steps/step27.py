import numpy as np
import math
from dezero import Variable, Function
from dezero.utils import get_dot_graph


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x):
    return Sin()(x)


x = Variable(np.array(np.pi / 4))
y = sin(x)
y.backward()
print('--- original sin ---')
print(y.data)
print(x.grad)


def my_sin(x, iters=5):
    y = 0
    for i in range(iters):
        t = (-1) ** i * x ** (2 * i + 1) / math.factorial(2 * i + 1)
        y = y + t
    return y


def my_sin(x, threshould=0.0001):
    y = 0
    for i in range(100000):
        t = (-1) ** i * x ** (2 * i + 1) / math.factorial(2 * i + 1)
        y = y + t
        if abs(t.data) < threshould:
            break
    return y


x = Variable(np.array(np.pi / 4))
y = my_sin(x)  # , threshould=1e-150)
y.backward()
print('--- approximate sin ---')
print(y.data)
print(x.grad)  # 0.7071032148228457

# 可視化 (dotファイルに保存)
x.name = 'x'
y.name = 'y'
dot = get_dot_graph(y)
with open('my_sin.dot', 'w') as o:
    o.write(dot)