import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import dezero
import dezero.functions as F
import dezero.layers as L
from dezero import DataLoader
from dezero.models import Model
from dezero.optimizers import Adam


use_gpu = dezero.cuda.gpu_enable
max_epoch = 10
batch_size = 16
latent_size = 2


class Encoder(Model):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.conv1 = L.Conv2d(32, kernel_size=3, stride=1, pad=1)
        self.conv2 = L.Conv2d(64, kernel_size=3, stride=2, pad=1)
        self.conv3 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv4 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.linear1 = L.Linear(32)
        self.linear2 = L.Linear(latent_size)
        self.linear3 = L.Linear(latent_size)

    def __call__(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.flatten(x)
        x = F.relu(self.linear1(x))
        z_mean = self.linear2(x)
        z_log_var = self.linear3(x)
        return z_mean, z_log_var

    def sampling(self, z_mean, z_log_var):
        batch_size = len(z_mean)
        xp = dezero.cuda.get_array_module(z_mean.data)
        epsilon = xp.random.randn(batch_size, self.latent_size)
        return z_mean + F.exp(z_log_var) * epsilon


class Decoder(Model):
    def __init__(self):
        super().__init__()
        self.to_shape = (64, 14, 14)  # (C, H, W)
        self.linear = L.Linear(np.prod(self.to_shape))
        self.deconv = L.Deconv2d(32, kernel_size=4, stride=2, pad=1)
        self.conv = L.Conv2d(1, kernel_size=3, stride=1, pad=1)

    def __call__(self, x):
        x = F.relu(self.linear(x))
        x = F.reshape(x, (-1,) + self.to_shape)  # reshape to (-1, C, H, W)
        x = F.relu(self.deconv(x))
        x = self.conv(x)
        x = F.sigmoid(x)
        return x


class VAE(Model):
    def __init__(self, latent_size):
        super().__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder()

    def __call__(self, x, C=1.0, k=1):
        """Call loss function of VAE.
        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            x (Variable or ndarray): Input variable.
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        z_mean, z_log_var = self.encoder(x)

        rec_loss = 0
        for l in range(k):
            z = self.encoder.sampling(z_mean, z_log_var)
            y = self.decoder(z)
            rec_loss += F.binary_cross_entropy(F.flatten(y), F.flatten(x)) / k

        kl_loss = C * (z_mean ** 2 + F.exp(z_log_var) - z_log_var - 1) * 0.5
        kl_loss = F.sum(kl_loss) / len(x)
        return rec_loss + kl_loss


def show_digits(epoch=0):
    """Display a 2D manifold of the digits"""
    n = 15  # 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            if use_gpu:
                z_sample = dezero.cuda.as_cupy(z_sample)
            with dezero.no_grad():
                x_decoded = vae.decoder(z_sample)
            if use_gpu:
                x_decoded.data = dezero.cuda.as_numpy(x_decoded.data)
            digit = x_decoded.data.reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    #plt.savefig('vae_{}.png'.format(epoch))


vae = VAE(latent_size)
optimizer = Adam().setup(vae)

transform = lambda x: (x / 255.0).astype(np.float32)
train_set = dezero.datasets.MNIST(train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size)

if use_gpu:
    vae.to_gpu()
    train_loader.to_gpu()
    xp = dezero.cuda.cupy
else:
    xp = np

for epoch in range(max_epoch):
    avg_loss = 0
    cnt = 0

    for x, t in train_loader:
        cnt += 1

        loss = vae(x)
        vae.cleargrads()
        loss.backward()
        optimizer.update()

        avg_loss += loss.data
        interval = 100 if use_gpu else 10
        if cnt % interval == 0:
            epoch_detail = epoch + cnt / train_loader.max_iter
            print('epoch: {:.2f}, loss: {:.4f}'.format(epoch_detail,
                                                       float(avg_loss/cnt)))

    show_digits(epoch)