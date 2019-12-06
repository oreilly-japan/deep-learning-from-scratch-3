import unittest
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check


class TestMSE_simple(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([0.0, 1.0, 2.0])
        x1 = np.array([0.0, 1.0, 2.0])
        expected = ((x0 - x1) ** 2).sum() / x0.size
        y = F.mean_squared_error_simple(x0, x1)
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward1(self):
        x0 = np.random.rand(10)
        x1 = np.random.rand(10)

        f = lambda x0: F.mean_squared_error_simple(x0, x1)
        self.assertTrue(gradient_check(f, x0))

    def test_backward2(self):
        x0 = np.random.rand(100)
        x1 = np.random.rand(100)

        f = lambda x0: F.mean_squared_error_simple(x0, x1)
        self.assertTrue(gradient_check(f, x0))


class TestMSE_simple(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([0.0, 1.0, 2.0])
        x1 = np.array([0.0, 1.0, 2.0])
        expected = ((x0 - x1) ** 2).sum() / x0.size
        y = F.mean_squared_error(x0, x1)
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward1(self):
        x0 = np.random.rand(10)
        x1 = np.random.rand(10)

        f = lambda x0: F.mean_squared_error(x0, x1)
        self.assertTrue(gradient_check(f, x0))

    def test_backward2(self):
        x0 = np.random.rand(100)
        x1 = np.random.rand(100)

        f = lambda x0: F.mean_squared_error(x0, x1)
        self.assertTrue(gradient_check(f, x0))