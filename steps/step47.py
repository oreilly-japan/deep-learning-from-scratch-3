import numpy as np

np.random.seed(0)  # シードを固定
from dezero import Variable, Model
import dezero.functions as F
from dezero.models import TwoLayerNet


def softmax1d(x):
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y


model = TwoLayerNet(in_size=2, hidden_size=10, out_size=3)

x = Variable(np.array([0.2, -0.4]))
y = model(x)
p = softmax1d(y)
print(y)  # variable([-0.6150578  -0.42790162  0.31733288])
print(p)  # variable([0.21068638 0.25404893 0.53526469])

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
y = model(x)
p = F.softmax(y)
print(y)
# variable([[-0.6150578  -0.42790162  0.31733288]
#  [-0.76395315 -0.24976449  0.18591381]
#  [-0.52006394 -0.96254613  0.57818937]
#  [-0.94252168 -0.50307479  0.17576322]])

print(p)
# variable([[0.21068638 0.25404893 0.53526469]
# [0.19019916 0.31806647 0.49173437]
# [0.21545395 0.13841619 0.64612986]
# [0.17820703 0.27655034 0.54524263]])

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
y = model(x)
loss = F.softmax_cross_entropy(y, t)
loss.backward()
print(loss)  # variable(1.49674426)