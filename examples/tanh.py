'''
y=tanh(x)のn階微分
計算グラフの可視化は、次のコマンドで行えます
$ python tanh.py | dot -T png -o sample.png
'''
import numpy as np
from dezero import Variable
from dezero.utils import get_dot_graph
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 3

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters + 1)
print(get_dot_graph(gx))