import unittest
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check


class TestMax(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.random.rand(10))
        y = F.max(x)
        expected = np.max(x.data)
        self.assertTrue(np.allclose(y.data, expected))

    def test_forward2(self):
        shape = (10, 20, 30)
        axis = 1
        x = Variable(np.random.rand(*shape))
        y = F.max(x, axis=axis)
        expected = np.max(x.data, axis=axis)
        self.assertTrue(np.allclose(y.data, expected))

    def test_forward3(self):
        shape = (10, 20, 30)
        axis = (0, 1)
        x = Variable(np.random.rand(*shape))
        y = F.max(x, axis=axis)
        expected = np.max(x.data, axis=axis)
        self.assertTrue(np.allclose(y.data, expected))

    def test_forward4(self):
        shape = (10, 20, 30)
        axis = (0, 1)
        x = Variable(np.random.rand(*shape))
        y = F.max(x, axis=axis, keepdims=True)
        expected = np.max(x.data, axis=axis, keepdims=True)
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward1(self):
        x_data = np.random.rand(10)
        f = lambda x: F.max(x)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.rand(10, 10) * 100
        f = lambda x: F.max(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 20, 30) * 100
        f = lambda x: F.max(x, axis=(1, 2))
        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.rand(10, 20, 20) * 100
        f = lambda x: F.sum(x, axis=None)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward5(self):
        x_data = np.random.rand(10, 20, 20) * 100
        f = lambda x: F.sum(x, axis=None, keepdims=True)
        self.assertTrue(gradient_check(f, x_data))