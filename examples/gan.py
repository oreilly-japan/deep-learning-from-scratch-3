# WIP

import numpy as np
import matplotlib.pyplot as plt
import dezero
import dezero.functions as F
import dezero.layers as L
from dezero import DataLoader
from dezero.models import Sequential
from dezero.optimizers import Adam


C, H, W = 512, 3, 3
G = Sequential(
    L.Linear(C*H*W),
    F.Reshape((-1, C, H, W)),
    L.BatchNorm(),
    F.ReLU(),
    L.Deconv2d(C // 2, kernel_size=2, stride=2, pad=1),
    L.BatchNorm(),
    F.ReLU(),
    L.Deconv2d(C // 4, kernel_size=2, stride=2, pad=1),
    L.BatchNorm(),
    F.ReLU(),
    L.Deconv2d(C // 8, kernel_size=2, stride=2, pad=1),
    L.BatchNorm(),
    F.ReLU(),
    L.Deconv2d(1, kernel_size=3, stride=3, pad=1),
    F.Sigmoid()
)

D = Sequential(
    L.Conv2d(64, kernel_size=3, stride=3, pad=1),
    F.LeakyReLU(0.1),
    L.Conv2d(128, kernel_size=2, stride=2, pad=1),
    L.BatchNorm(),
    F.LeakyReLU(0.1),
    L.Conv2d(256, kernel_size=2, stride=2, pad=1),
    L.BatchNorm(),
    F.LeakyReLU(0.1),
    L.Conv2d(512, kernel_size=2, stride=2, pad=1),
    L.BatchNorm(),
    F.LeakyReLU(0.1),
    F.flatten,
    L.Linear(1)
)

D.layers[0].W.name = 'conv1_W'
D.layers[0].b.name = 'conv1_b'

def init_weight(D, G, hidden_size):
    # dummy data
    batch_size = 1
    z = np.random.rand(batch_size, hidden_size)
    fake_images = G(z)
    D(fake_images)

    for l in D.layers + G.layers:
        classname = l.__class__.__name__
        if 'conv' in classname.lower():
            l.W.data = 0.02 * np.random.randn(*l.W.data.shape)


max_epoch = 100
batch_size = 100
hidden_size = 1000

init_weight(D, G, hidden_size)

train_set = dezero.datasets.MNIST(train=True,
                                  transforms=dezero.transforms.AsType('f'))
train_loader = DataLoader(train_set, batch_size)

label_real = np.ones(batch_size).astype(np.int)
label_fake = np.zeros(batch_size).astype(np.int)

if dezero.cuda.gpu_enable:
    G.to_gpu()
    D.to_gpu()
    train_loader.to_gpu()
    label_real = dezero.cuda.as_cupy(label_real)
    label_fake = dezero.cuda.as_cupy(label_fake)

opt_g = Adam(alpha=0.0002, beta1=0.5).setup(G)
opt_g.add_hook(dezero.optimizers.WeightDecay(0.0001))
opt_d = Adam(alpha=0.0002, beta1=0.5).setup(D)
opt_d.add_hook(dezero.optimizers.WeightDecay(0.0001))


test_z = np.random.randn(25, hidden_size).astype(np.float32)

def generate_image(epoch=0):
    fake_images = G(test_z)
    fig = plt.figure()
    for i in range(0, 25):
        ax = plt.subplot(5, 5, i+1)
        ax.axis('off')
        plt.imshow(fake_images.data[i][0], 'gray')
    plt.savefig('gan_{}.png'.format(epoch))

generate_image()

for epoch in range(max_epoch):
    for x, t in train_loader:
        # real
        y_real = D(x)

        # fake
        z = np.random.randn(batch_size, hidden_size).astype(np.float32)
        x_fake = G(z)
        y_fake = D(x_fake)
        d_loss = F.sigmoid_cross_entropy(y_real, label_real) + \
               F.sigmoid_cross_entropy(y_fake, label_fake)

        G.cleargrads()
        D.cleargrads()
        d_loss.backward()
        opt_d.update()

        # train for Generator
        z = np.random.randn(batch_size, hidden_size).astype(np.float32)
        x_fake = G(z)
        y_fake = D(x_fake)
        g_loss = F.sigmoid_cross_entropy(y_fake, label_real)

        G.cleargrads()
        D.cleargrads()
        g_loss.backward()
        opt_g.update()

        print(g_loss, d_loss)
    generate_image(epoch)
