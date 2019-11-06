import numpy as np
np.random.seed(0)

from chainer import Chain
from chainer.optimizers import SGD
import chainer.links as L

from dezero.iterators import RnnIterator
from dezero.datasets import load_shakespear


class SimpleRNN(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()

        I, H, O = in_size, hidden_size, out_size
        with self.init_scope():
            self.embed = L.EmbedID(I, H)
            #self.rnn = L.RNN(H, H)
            self.rnn = L.LSTM(H, H)
            self.h2y = L.Linear(H, O)

    def reset_state(self):
        self.rnn.reset_state()

    def __call__(self, x):
        e = self.embed(x)
        h = self.rnn(e)
        y = self.h2y(h)
        return y


max_epoch = 100
hidden_size = 512#100
bptt_length = 30
batch_size = 100
lr = 0.05#1e-4

indices, char_to_id, id_to_char = load_shakespear()
iterator = RnnIterator(indices, batch_size)
vocab_size = len(char_to_id)
rnn = SimpleRNN(vocab_size, hidden_size, vocab_size)
model = L.Classifier(rnn)
optimizer = SGD(lr=lr)
optimizer.setup(model)

def generate_sample(n=30, init_char=' '):
    rnn.reset_state()

    s = ''
    x = np.array([char_to_id[init_char]])
    for i in range(n):
        y = rnn(x)
        m = y.data.argmax()
        c = id_to_char[m]
        s += c
        x = np.array([m])

    rnn.reset_state()
    return s

loss, count = 0, 0
while iterator.epoch < max_epoch:
    batch_x, batch_t = iterator.next()
    loss += model(batch_x, batch_t)

    count += 1
    if count % bptt_length == 0:
        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

    if count % 100 == 0:
        avg_loss = float(loss.data) / count
        print('| iters %d | loss %f' % (count, avg_loss))

    if iterator.is_new_epoch:
        print(generate_sample(300, init_char='\n'))
        print('epoch:', iterator.epoch)
        loss, count = 0, 0
