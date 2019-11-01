import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
c = Variable(np.array([[0, 10, 20], [30, 40, 50]]))
t = x + c
y = F.sum(t)

y.backward(retain_grad=True)
print(y.grad.data)
print(t.grad.data)
print(x.grad.data)
print(c.grad.data)