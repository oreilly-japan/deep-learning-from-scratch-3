import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.models import TwoLayerNet

# データセットの生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

# ハイパーパラメータの設定
lr = 0.2
max_iter = 10000
hidden_size = 10

# モデルの生成
N, I = x.shape
N, O = y.shape
model = TwoLayerNet(I, hidden_size, O)

# 学習の開始
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    print(loss)