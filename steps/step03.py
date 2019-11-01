import numpy as np


class Variable:

    def __init__(self, data):
        self.data = data


class Function:

    def __call__(self, input):
        x = input.data
        y = self.forward(x)  # 具体的な計算はforward()で行う
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):

    def forward(self, x):
        return x ** 2


class Exp(Function):

    def forward(self, x):
        return np.exp(x)


f = Square()
g = Exp()
h = Square()

x = Variable(np.array(0.5))
y = f(x)
z = g(y)
a = h(z)
print(a.data)