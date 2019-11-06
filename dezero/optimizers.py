import math
from dezero import cuda


class Optimizer:
    def setup(self, link):
        self.target = link
        return self

    def update(self):
        for param in self.target.params():
            self.update_one(param)

    def update_one(self, param):
        NotImplementedError()


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        h = self.hs[h_key]

        h += grad * grad
        param.data -= lr * grad / (xp.sqrt(h) + eps)


class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6):
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.msg:
            self.msg[key] = xp.zeros_like(param.data)
            self.msdx[key] = xp.zeros_like(param.data)

        msg, msdx = self.msg[key], self.msdx[key]
        rho = self.rho
        eps = self.eps
        grad = param.grad.data

        msg *= rho
        msg += (1 - rho) * grad * grad
        dx = xp.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (xp.sqrt(v) + eps)