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
    __array_priority__ = 200

    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
        self.priority = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def array(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return '%s(%s)' % ('variable', np.array2string(self.data))

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
            gys = [output.grad for output in f.outputs]
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
                    y.cleargrad()


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    elif isinstance(obj, np.ndarray):
        return Variable(obj)
    return Variable(np.array(obj))


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]

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
            self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    f = Add()
    y = f(x0, x1)
    return y


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    f = Mul()
    return f(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    f = Neg()
    return f(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy * 1, gy * -1


def sub(x0, x1):
    f = Sub()
    return f(x0, x1)


def rsub(x0, x1):
    return sub(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    f = Div()
    return f(x0, x1)


def rdiv(x0, x1):
    return div(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c

        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    f = Pow(c)
    return f(x)


Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow

x = Variable(np.array(2.0))
y = -x
y.backward()
print(y)  # variable(-2.)
print(x.grad)  # -1.0

x = Variable(np.array(2.0))
y = 2.0 - x
y.backward()
print(y)  # variable(0.)

x = Variable(np.array(2.0))
y = 3.0 / x
y.backward()
print(y)  # variable(1.5)

x = Variable(np.array(3.0))
y = x ** 3
y.backward()
print(y)  # variable(27)