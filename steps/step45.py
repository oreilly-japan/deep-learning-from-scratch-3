import numpy as np
from dezero import Variable, Model
import dezero.layers as L
import dezero.functions as F


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

# ハイパーパラメータの設定
lr = 0.2
max_iter = 10000
hidden_size = 10

# モデルの生成
class TwoLayerNet(Model):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(in_size, hidden_size)
        self.l2 = L.Linear(hidden_size, out_size)

    def __call__(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(1, hidden_size, 1)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    print(loss)