import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.random.randn(2, 3))
w = Variable(np.random.randn(3, 4))
t = F.matmul(x, w)
y.backward()

print(x.grad.shape)
print(w.grad.shape)