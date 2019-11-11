import weakref
import numpy as np
import contextlib


class Config:
    enable_backprop = True


config = Config()


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(config, name)
    setattr(config, name, value)
    try:
        yield
    finally:
        setattr(config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
        self.priority = 0

    def set_creator(self, func):
        self.creator = func
        self.priority = func.priority + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.priority)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # output is weakref
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y is weakref


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]

        if config.enable_backprop:
            self.priority = max([x.priority for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy


def square(x):
    return Square()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)
    print(y.data)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)
    print(y.data)

x = Variable(np.array(2.0))
y = square(x)
y.backward()
print(y.data)
print(x.grad)