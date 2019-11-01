import numpy as np
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import TwoLayerNet

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

lr = 0.2
max_iter = 10000
hidden_size = 10

N, I = x.shape
N, O = y.shape
model = TwoLayerNet(I, hidden_size, O)
optimizer = optimizers.SGD(lr).setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    print(loss)

# グラフの描画
import matplotlib.pyplot as plt

plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = Variable(np.arange(0, 1, .01)[:, np.newaxis])
y_pred = model(t)
plt.plot(t.data, y_pred.data, color='r')
plt.show()