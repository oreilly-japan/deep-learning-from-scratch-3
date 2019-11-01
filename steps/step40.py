import numpy as np
from dezero import Variable
import dezero.functions as F

x0 = Variable(np.array([1, 2, 3]))
x1 = Variable(np.array([10]))
z = x0 + x1
# y = z.sum()
y = F.sum(z)
print(y)

y.backward()
print(x1.grad)