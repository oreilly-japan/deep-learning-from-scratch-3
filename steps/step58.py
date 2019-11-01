import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import Variable, Chain
from dezero.optimizers import SGD
import dezero.functions as F
import dezero.layers as L

np.random.seed(0)


class SimpleRNN(Chain):

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()

        I, H, O = in_size, hidden_size, out_size
        with self.init_scope():
            self.x2h = L.Linear(I, H)
            self.h2h = L.Linear(H, H)
            self.h2y = L.Linear(H, O)

        self.h = None

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))

        y = self.h2y(h_new)
        self.h = h_new
        return y

# データ
def get_dataset(N):
    x = np.linspace(0, 2 * np.pi, N)
    noise_range = (-0.05, 0.05)
    noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
    y = np.sin(x) + noise
    return y

max_epoch = 20
batch_size = 30
hidden_size = 100
bptt_length = 30

rnn = SimpleRNN(1, hidden_size, 1)
model = L.Classifier(rnn, F.mean_squared_error, None)
optimizer = SGD(lr=1e-4)
optimizer.setup(model)

# 学習データ
train_size = 1000
x_list = get_dataset(train_size)
xs = x_list[:-1]
ts = x_list[1:]
seqlen = len(xs)

for epoch in range(max_epoch):
    rnn.reset_state()

    loss, count = 0, 0

    for x, t in zip(xs, ts):
        loss += model(x, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()

            #loss.unchain_backward()
            loss.unchain()
            rnn.h.unchain()

            optimizer.update()

    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1,avg_loss))


xs = np.cos(np.linspace(0, 10 * np.pi, 1000))

# グラフの描画
rnn.reset_state()
pred_list = []
with dezero.no_grad():
    for x in xs:
        y = rnn(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='train data')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
