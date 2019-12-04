import numpy as np
import dezero.functions as F
from dezero.core import Parameter
from dezero.utils import _pair


# =============================================================================
# Layer / Model
# =============================================================================
class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, data, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(data, key)
            else:
                data[key] = obj.data

    def save_weights(self, path):
        self.to_cpu()
        data = {}
        self._flatten_params(data)
        np.savez_compressed(path, **data)

    def load_weights(self, path):
        npz = np.load(path)
        data = {}
        self._flatten_params(data)
        for key, param in data.items():
            param[...] = npz[key]


# alias
class Model(Layer):
    pass


# =============================================================================
# Linear / Conv / EmbedID / RNN / LSTM
# =============================================================================
class Linear(Layer):
    def __init__(self, in_size, out_size, nobias=False):
        super().__init__()

        I, O = in_size, out_size
        W_data = np.random.randn(I, O).astype(np.float32) * np.sqrt(1 / I)
        self.W = Parameter(W_data, name='W')
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(O, dtype=np.float32), name='b')

    def __call__(self, x):
        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False):
        super().__init__()
        self.stride = stride
        self.pad = pad

        C, OC = in_channels, out_channels
        KH, KW = _pair(kernel_size)

        W_data = np.random.randn(OC, C, KH, KW).astype(np.float32) * np.sqrt(
            1 / C * KH * KW)
        self.W = Parameter(W_data, name='W')
        if nobias:
            self.b = None
        else:
            b_data = np.zeros(OC).astype(np.float32)
            self.b = Parameter(b_data, name='b')

    def __call__(self, x):
        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class EmbedID(Layer):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name='W')

    def __call__(self, x):
        y = self.W[x]
        return y


class RNN(Layer):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        I, H = in_size, hidden_size
        self.x2h = Linear(I, H)
        self.h2h = Linear(H, H)
        self.h = None

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))

        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(self, in_size, hidden_size):
        super().__init__()

        I, H = in_size, hidden_size
        self.x2f = Linear(I, H)
        self.x2i = Linear(I, H)
        self.x2o = Linear(I, H)
        self.x2u = Linear(I, H)
        self.h2f = Linear(H, H, nobias=True)
        self.h2i = Linear(H, H, nobias=True)
        self.h2o = Linear(H, H, nobias=True)
        self.h2u = Linear(H, H, nobias=True)

        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def __call__(self, x):
        if self.h is None:
            N, D = x.shape
            H, H = self.h2f.W.shape
            self.h = np.zeros((N, H), np.float32)
            self.c = np.zeros((N, H), np.float32)

        f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
        i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
        o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
        u = F.tanh(self.x2u(x) + self.h2u(self.h))

        c = (f * self.c) + (i * u)
        h = o * F.tanh(c)

        self.h, self.c = h, c
        return h