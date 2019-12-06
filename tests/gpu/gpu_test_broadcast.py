import unittest
import cupy as np  # !! CUPY !!
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check


class TestBroadcast(unittest.TestCase):

    def test_shape_check(self):
        x = Variable(np.random.randn(1, 10))
        b = Variable(np.random.randn(10))
        y = x + b
        loss = F.sum(y)
        loss.backward()
        self.assertEqual(b.grad.shape, b.shape)