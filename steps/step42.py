import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F
import dezero

# トイ・データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

N, I = x.shape
N, O = y.shape
W = Variable(np.zeros((I, O)))
b = Variable(np.zeros(O))


def predict(x):
    y = F.matmul(x, W) + b
    return y


def mean_squared_error(y1, y2):
    N = y1.shape[0]
    diff = y1 - y2
    loss = F.sum(diff * diff) / N
    return loss


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    with dezero.no_grad():
        W -= W.grad.data * lr
        b -= b.grad.data * lr
    # W.data -= lr * W.grad
    # b.data -= lr * b.grad
    print(loss)

'''
# グラフの描画
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
y_pred = predict(x)
plt.plot(x.data, y_pred.data, color='r')
plt.show()
'''