import numpy as np
from dezero import utils
from dezero.core import Function, as_variable
from dezero.utils import _pair
from dezero.functions import linear


# =============================================================================
# conv2d_simple / pooling_simple (simple version)
# =============================================================================
def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)

    n, c, h, w = x.shape
    out_c, c, kh, kw = W.shape
    sh, sw = _pair(stride)
    ph, pw = _pair(pad)
    out_h = utils.get_conv_outsize(h, kh, sh, ph)
    out_w = utils.get_conv_outsize(w, kw, sw, pw)

    col = im2col(x, (kh, kw), stride, pad)

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((n * out_h * out_w, -1))
    W = W.reshape((out_c, -1)).transpose()

    t = linear(col, W, b)

    y = t.reshape((n, out_h, out_w, -1)).transpose((0, 3, 1, 2))
    return y


def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)

    n, c, h, w = x.shape
    kh, kw = _pair(kernel_size)
    ph, pw = _pair(pad)
    sh, sw = _pair(stride)
    out_h = (h + ph * 2 - kh) // sh + 1
    out_w = (h + pw * 2 - kw) // sw + 1

    col = im2col(x, kernel_size, stride, pad)

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((-1, kh * kw))

    y = col.max(axis=1)

    y = y.reshape((n, out_h, out_w, c))
    y = y.transpose((0, 3, 1, 2))
    return y


# =============================================================================
#  conv2d / deconv2d / pooling
# =============================================================================
class Conv2d(Function):

    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = _pair(stride)
        self.pad = _pair(pad)

    def forward(self, x, W, b):
        kh, kw = W.shape[2:]
        col = utils.im2col(x, (kh, kw), self.stride, self.pad)

        y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = np.rollaxis(y, 3, 1)
        # y = np.transpose(y, (0, 3, 1, 2))
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        # ==== gx ====
        gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(x, gy)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    f = Conv2d(stride, pad)
    return f(x, W, b)


class Deconv2d(Function):

    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        sy, sx = self.stride
        ph, pw = self.pad
        in_c, out_c, kh, kw = W.shape
        n, in_c, in_h, in_w = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(in_h, kh, sy, ph)
            out_w = get_deconv_outsize(in_w, kw, sx, pw)
        else:
            out_h, out_w = _pair(self.outsize)
        img_shape = (n, out_c, out_h, out_w)

        gcol = np.tensordot(W, x, (0, 1))
        gcol = np.rollaxis(gcol, 3)
        y = utils.col2im(gcol, img_shape, (kh, kw), self.stride, self.pad)
        # b, k, h, w
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    f = Deconv2d(stride, pad, outsize)
    return f(x, W, b)


class Conv2DGradW(Function):

    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        col = utils.im2col(x, self.kernel_size, self.stride, self.pad)
        gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


class Pooling(Function):

    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        col = utils.im2col(x, self.kernel_size, self.stride, self.pad)

        n, c, kh, kw, out_h, out_w = col.shape
        col = col.reshape(n, c, kh * kw, out_h, out_w)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        f = Pooling2DGrad(self)
        return f(gy)


class Pooling2DGrad(Function):

    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        n, c, out_h, out_w = gy.shape
        h, w = self.input_shpae[2:]
        kh, kw = _pair(self.kernel_size)

        gcol = np.zeros((n * c * out_h * out_w * kh * kw), dtype=self.dtype)

        indexes = self.indexes.ravel() + np.arange(
            0, self.indexes.size * kh * kw, kh * kw)

        gcol[indexes] = gy[0].ravel()
        gcol = gcol.reshape(n, c, out_h, out_w, kh, kw)
        gcol = np.swapaxes(gcol, 2, 4)
        gcol = np.swapaxes(gcol, 3, 5)

        gx = utils.col2im(gcol, (n, c, h, w), self.kernel_size, self.stride,
                          self.pad)  # self.sy, self.sx, self.ph, self.pw, h, w)
        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):

    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = utils.im2col(x, self.kernel_size, self.stride, self.pad)
        n, c, kh, kw, out_h, out_w = col.shape
        col = col.reshape(n, c, kh * kw, out_h, out_w)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, kh * kw)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(n, c, out_h, out_w)


def pooling(x, kernel_size, stride=1, pad=0):
    f = Pooling(kernel_size, stride, pad)
    return f(x)


# =============================================================================
#  im2col / col2im
# =============================================================================
class Im2col(Function):

    def __init__(self, kernel_size, stride, pad):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        self.input_shape = x.shape
        y = utils.im2col(x, self.kernel_size, self.stride, self.pad)
        return y

    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride,
                    self.pad)
        return gx


def im2col(x, kernel_size, stride=1, pad=0):
    f = Im2col(kernel_size, stride, pad)
    return f(x)


class Col2im(Function):

    def __init__(self, input_shape, kernel_size, stride, pad):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        y = utils.col2im(x, self.input_shape, self.kernel_size, self.stride,
                         self.pad)
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0):
    f = Col2im(input_shape, kernel_size, stride, pad)
    return f(x)


def get_deconv_outsize(size, k, s, p):
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size, kernel_size, stride, pad):
    i, k, s, p = input_size, kernel_size, stride, pad
    return (i + p * 2 - k) // s + 1