import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP

max_epoch = 20
batch_size = 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True)
model = MLP((hidden_size, 10))
optimizer = optimizers.SGD().setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # Shuffle index for data
    index = np.random.permutation(data_size)

    for i in range(max_iter):
        # Create minibatch
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        samples = [train_set[i] for i in batch_index]
        x = np.array([sample[0] for sample in samples])
        t = np.array([sample[1] for sample in samples])

        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        if i % 100 == 0:
            print('loss', loss.data)