import unittest
import cupy as np  # !! CUPY !!
import dezero
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check, array_equal


class TestDropout(unittest.TestCase):

    def test_forward1(self):
        x = np.random.randn(100, 100)
        y = F.dropout(Variable(x), dropout_ratio=0.0)
        res = array_equal(y.data, x)
        self.assertTrue(res)

    def test_forward2(self):
        x = np.random.randn(100, 100)
        with dezero.test_mode():
            y = F.dropout(x)
        res = array_equal(y.data, x.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.random.randn(10, 10)

        def f(x):
            np.random.seed(0)
            return F.dropout(x, 0.5)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.randn(10, 20)

        def f(x):
            np.random.seed(0)
            return F.dropout(x, 0.99)

        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.randn(10, 10)

        def f(x):
            np.random.seed(0)
            return F.dropout(x, 0.0)

        self.assertTrue(gradient_check(f, x_data))