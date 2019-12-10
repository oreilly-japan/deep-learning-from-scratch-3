import math
import numpy as np
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP

max_epoch = 20
batch_size = 100
hidden_size = 1000

train_set, test_set = dezero.datasets.get_mnist()
model = MLP((hidden_size, 10))
optimizer = optimizers.SGD().setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # Shuffle data
    np.random.shuffle(train_set)

    for i in range(max_iter):
        # Create minibatch
        batch = train_set[i * batch_size:(i + 1) * batch_size]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        if i % 100 == 0:
            print('loss', loss.data)