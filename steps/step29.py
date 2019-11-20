import numpy as np
# core_simple を明示的にインポート
# （dezero/__init__.py の is_simple_core = False でも動作させるため）
from dezero.core_simple import Variable
from dezero.core_simple import setup_variable
setup_variable()


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


logs = []
x = Variable(np.array(2.0))
iters = 200

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= 0.01 * x.grad