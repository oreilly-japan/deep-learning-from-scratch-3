import unittest
import cupy as np  # !! CUPY !!
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check, array_equal
from dezero import utils


class TestIm2col(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 1, 3, 3
        x = np.arange(n * c * h * w).reshape((n, c, h, w))
        y = F.im2col(x, 3, 3, 0, to_matrix=True)
        expected = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])

        res = array_equal(y.data, expected)
        self.assertTrue(res)

    def test_backward1(self):
        n, c, h, w = 1, 1, 3, 3
        x = np.arange(n * c * h * w).reshape((n, c, h, w))
        f = lambda x: F.im2col(x, 3, 3, 0, to_matrix=True)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        n, c, h, w = 1, 1, 3, 3
        x = np.arange(n * c * h * w).reshape((n, c, h, w))
        f = lambda x: F.im2col(x, 3, 3, 0, to_matrix=False)
        self.assertTrue(gradient_check(f, x))


class TestCol2in(unittest.TestCase):

    def test_backward1(self):
        n, c, h, w = 1, 1, 3, 3
        x = np.random.rand(1, 9)
        f = lambda x: F.col2im(x, (n, c, h, w), 3, 3, 0, to_matrix=True)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        n, c, h, w = 1, 1, 3, 3
        x = np.random.rand(1, 1, 3, 3, 1, 1)
        f = lambda x: F.col2im(x, (n, c, h, w), 3, 3, 0, to_matrix=False)
        self.assertTrue(gradient_check(f, x))