import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.dataset import DatasetLoader
from dezero.models import TwoLayerNet

max_epoch = 20
batch_size = 100
hidden_size = 1000

train_set, test_set = dezero.datasets.get_mnist()
train_loader = DatasetLoader(train_set, batch_size)
test_loader = DatasetLoader(test_set, batch_size, shuffle=False)

model = TwoLayerNet(784, hidden_size, 10)
optimizer = optimizers.SGD().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc) * len(t)

    print('epoch: {}'.format(epoch))
    print('train loss: {}, accuracy: {}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc) * len(t)

    print('test loss: {}, accuracy: {}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))