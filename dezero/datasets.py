import gzip
import numpy as np
import matplotlib.pyplot as plt
from dezero.utils import get_file
from dezero.data import Dataset, TupleDataset
from dezero.transforms import Compose, Flatten, ToFloat, Normalize


class Spiral(Dataset):

    def prepare(self):
        seed = 1984 if self.train else 2020
        np.random.seed(seed=seed)

        num_data, num_class, input_dim = 100, 3, 2
        data_size = num_class * num_data
        x = np.zeros((data_size, input_dim), dtype=np.float32)
        t = np.zeros(data_size, dtype=np.int)

        for j in range(num_class):
            for i in range(num_data):
                rate = i / num_data
                radius = 1.0 * rate
                theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
                ix = num_data * j + i
                x[ix] = np.array([radius * np.sin(theta),
                                  radius * np.cos(theta)]).flatten()
                t[ix] = j
        # Shuffle
        indices = np.random.permutation(num_data * num_class)
        x = x[indices]
        t = t[indices]
        self.data, self.label = x, t


class MNIST(Dataset):

    def __init__(self, train=True,
                 transforms=Compose([Flatten(), ToFloat(),
                                     Normalize(0., 255.)]),
                 target_transforms=None):
        super().__init__(train, transforms, target_transforms)

    def prepare(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                8: '8', 9: '9'}


class SinCurve(Dataset):

    def prepare(self):
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]


class ImageNet(Dataset):

    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = get_file(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels


def get_shakespear():
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    file_name = 'shakespear.txt'
    path = get_file(url, file_name)

    with open(path, 'r') as f:
        data = f.read()
    chars = list(data)

    char_to_id = {}
    id_to_char = {}
    for word in data:
        if word not in char_to_id:
            new_id = len(char_to_id)
            char_to_id[word] = new_id
            id_to_char[new_id] = word

    indices = np.array([char_to_id[c] for c in chars])
    return indices, char_to_id, id_to_char