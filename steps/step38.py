import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
y = F.reshape(x, (6,))
# or y = x.reshape(6)
y.backward(retain_grad=True)
print(y.grad)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
# or y = x.T
y.backward()
print(x.grad)


