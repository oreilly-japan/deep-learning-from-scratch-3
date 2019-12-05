import numpy as np
import os.path
import gzip
import os
from dezero.utils import download_cache, cache_dir


def get_spiral(seed=1984):
    np.random.seed(seed)
    N = 200  # クラスごとのサンプル数
    DIM = 2  # データの要素数
    CLS_NUM = 3  # クラス数
    TRAIN_SIZE = 100  # 訓練データのサイズ

    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM), dtype=np.int)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j

    # shuffle
    indices = np.random.permutation(N*CLS_NUM)
    x = x[indices]
    t = t[indices]

    # tupled dataset
    train_size = TRAIN_SIZE*CLS_NUM
    train_set = [(x, t) for x, t in zip(x[:train_size], t[:train_size])]
    test_set = [(x, t) for x, t in zip(x[:train_size], t[train_size:])]
    return train_set, test_set


class MNIST:
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_label': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_label': 't10k-labels-idx1-ubyte.gz'
    }
    train_num = 60000
    test_num = 10000
    img_dim = (1, 28, 28)
    img_size = 784
    save_path = os.path.join(cache_dir, "mnist.npz")

    def download_mnist(self):
        for v in MNIST.key_file.values():
            download_cache(MNIST.url_base + v)

    def _load_label(self, file_name):
        file_path = os.path.join(cache_dir, file_name)

        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return labels

    def _load_img(self, file_name):
        file_path = os.path.join(cache_dir, file_name)

        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, MNIST.img_size)

        return data

    def _convert_numpy(self):
        key_file = MNIST.key_file
        x0 = self._load_img(key_file['train_img'])
        t0 = self._load_label(key_file['train_label'])
        x1 = self._load_img(key_file['test_img'])
        t1 = self._load_label(key_file['test_label'])
        return x0, t0, x1, t1

    def init_mnist(self):
        self.download_mnist()
        x0, t0, x1, t1 = self._convert_numpy()
        np.savez_compressed(MNIST.save_path, x0=x0, t0=t0, x1=x1, t1=t1)
        return x0, t0, x1, t1

    def preprocess(self, x, ndim, scale, dtype):
        _shape = {1: (-1, 784), 2: (-1, 28, 28), 3: (-1, 1, 28, 28)}

        x = x.astype(dtype)
        x /= (255.0 / scale)
        x = x.reshape(*_shape[ndim])
        return x

    def get_mnist(self, ndim=1, scale=1.0, dtype=np.float32):
        if not os.path.exists(MNIST.save_path):
            self.init_mnist()

        D = np.load(MNIST.save_path)
        x_train, t_train, x_test, t_test = D['x0'], D['t0'], D['x1'], D['t1']

        x_train = self.preprocess(x_train, ndim, scale, dtype)
        x_test = self.preprocess(x_test, ndim, scale, dtype)

        train = [(x_train[i], t_train[i]) for i in range(len(t_train))]
        test = [(x_test[i], t_test[i]) for i in range(len(t_test))]
        return train, test


def get_mnist(ndim=1, scale=1.0, dtype=np.float32):
    m = MNIST()
    return m.get_mnist(ndim, scale, dtype)


def load_shakespear():
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    file_name = 'shakespear.txt'
    download_cache(url, file_name)

    with open(os.path.join(cache_dir, file_name), 'r') as f:
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