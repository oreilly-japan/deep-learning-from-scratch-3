import time
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.dataset import DatasetLoader
from dezero.models import MLP

max_epoch = 3
batch_size = 100

train_set, test_set = dezero.datasets.get_mnist()
train_loader = DatasetLoader(train_set, batch_size)
model = MLP((784, 1000, 10))
optimizer = optimizers.SGD().setup(model)

# GPU mode
train_loader.to_gpu()
model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    elapsed_time = time.time() - start
    print('epoch: {}, loss: {:.4f}, time: {:.4f}[sec]'.format(
        epoch + 1, sum_loss / len(train_set), elapsed_time))