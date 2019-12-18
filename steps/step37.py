import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
c = Variable(np.array([[0, 10, 20], [30, 40, 50]]))
t = x + c
y = F.sum(t)

y.backward(retain_grad=True)
print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)