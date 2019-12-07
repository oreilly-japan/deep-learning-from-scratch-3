import unittest
import cupy as np  # !! CUPY !!
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check, array_allclose


class TestMatmul(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        w = Variable(x.data.T)
        y = F.matmul(x, w)
        res = y.data
        expected = np.array([[14, 32], [32, 77]])
        self.assertTrue(array_allclose(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 2)
        w = np.random.randn(2, 3)
        f = lambda x: F.matmul(x, Variable(w))
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x_data = np.random.randn(10, 1)
        w_data = np.random.randn(1, 5)
        f = lambda w: F.matmul(Variable(x_data), w)
        self.assertTrue(gradient_check(f, w_data))