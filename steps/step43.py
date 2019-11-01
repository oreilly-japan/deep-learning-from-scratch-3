import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

# トイ・データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

N, I = x.shape
N, O = y.shape
H = 10

w1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
w2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))


def predict(x):
    y = F.matmul(x, w1) + b1
    y = F.sigmoid(y)
    y = F.matmul(y, w2) + b2
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    w1.cleargrad()
    b1.cleargrad()
    w2.cleargrad()
    b2.cleargrad()
    loss.backward()

    w1.data -= lr * w1.grad.data
    b1.data -= lr * b1.grad.data
    w2.data -= lr * w2.grad.data
    b2.data -= lr * b2.grad.data
    print(loss)

# グラフの描画
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')

t = Variable(np.arange(0, 1, .01)[:, np.newaxis])
y_pred = predict(t)
plt.plot(t.data, y_pred.data, color='r')
plt.show()