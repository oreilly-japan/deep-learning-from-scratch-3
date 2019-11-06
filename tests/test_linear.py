import unittest
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import check_backward


class TestLinear(unittest.TestCase):
    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        w = Variable(x.data.T)
        b = None
        y = F.linear(x, w, b)

        res = y.data
        expected = np.array([[14, 32], [32, 77]])
        self.assertTrue(np.array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 2)
        W = np.random.randn(2, 3)
        b = np.random.randn(3)
        f = lambda x: F.linear(x, W, b)
        self.assertTrue(check_backward(f, x))

    def test_backward1(self):
        x = np.random.randn(3, 2)
        W = np.random.randn(2, 3)
        b = np.random.randn(3)
        f = lambda x: F.linear(x, W, b)
        self.assertTrue(check_backward(f, x))

    def test_backward2(self):
        x = np.random.randn(100, 200)
        W = np.random.randn(200, 300)
        b = None
        f = lambda x: F.linear(x, W, b)
        self.assertTrue(check_backward(f, x))