import unittest
import numpy as np
from dezero import Variable
from dezero.utils import check_backward


class TestAdd(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 + x1

        res = y.data
        expected = np.array([2, 4, 6])
        self.assertTrue(np.array_equal(res, expected))

    def test_backward1(self):
        x = np.random.randn(3, 3)
        w = np.random.randn(3, 3)
        f = lambda x: x + Variable(w)
        self.assertTrue(check_backward(f, x))