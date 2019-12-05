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

    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        np.savez_compressed(path, **array_dict)

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# alias
class Model(Layer):
    pass


# =============================================================================
# Linear / Conv / EmbedID / RNN / LSTM
# =============================================================================
class Linear_simple(Layer):
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


class Linear(Layer):
    def __init__(self, in_size, out_size=None, nobias=False):
        super().__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.in_size = in_size
        self.out_size = out_size

        self.W = Parameter(None, name='W')
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=np.float32), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(np.float32) * np.sqrt(1 / I)
        self.W.data = W_data

    def __call__(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False):
        """

        Parameters
        ----------
        in_channels : int or None
            入力データのチャンネル数。Noneの場合はforward時のxからin_channelsを取得する
        out_channels : int
        kernel_size : int or (int, int)
        stride : int or (int, int)
        pad : int or (int, int)
        nobias : bool
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        self.W = Parameter(None, name='W')
        if nobias:
            self.b = None
        else:
            b_data = np.zeros(out_channels).astype(np.float32)
            self.b = Parameter(b_data, name='b')

    def _init_W(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = _pair(self.kernel_size)
        W_data = np.random.randn(OC, C, KH, KW).astype(np.float32) * np.sqrt(
            1 / C * KH * KW)
        self.W.data = W_data

    def __call__(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            self._init_W()

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