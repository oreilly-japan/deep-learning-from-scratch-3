import unittest
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import check_backward


class TestDropout(unittest.TestCase):

    def test_forward1(self):
        x = np.random.randn(100, 100)
        y = F.dropout(Variable(x), dropout_ratio=0.0)
        res = np.array_equal(y.data, x.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.random.randn(10, 10)

        def f(x):
            np.random.seed(0)
            return F.dropout(x, 0.5)

        self.assertTrue(check_backward(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 20)

        def f(x):
            np.random.seed(0)
            return F.dropout(x, 0.99)

        self.assertTrue(check_backward(f, x_data))

    def test_backward3(self):
        x_data = np.random.randn(10, 10)

        def f(x):
            np.random.seed(0)
            return F.dropout(x, 0.0)

        self.assertTrue(check_backward(f, x_data))