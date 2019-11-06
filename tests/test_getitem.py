import unittest
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import check_backward


class TestGetitem(unittest.TestCase):
    def test_forward1(self):
        x_data = np.arange(12).reshape((2, 2, 3))
        x = Variable(x_data)
        y = F.get_item(x, 0)
        self.assertTrue(np.allclose(y.data, x_data[0]))

    def test_forward1a(self):
        x_data = np.arange(12).reshape((2, 2, 3))
        x = Variable(x_data)
        y = x[0]
        self.assertTrue(np.allclose(y.data, x_data[0]))

    def test_forward2(self):
        x_data = np.arange(12).reshape((2, 2, 3))
        x = Variable(x_data)
        y = F.get_item(x, (0, 0, slice(0, 2, 1)))
        self.assertTrue(np.allclose(y.data, x_data[0, 0, 0:2:1]))

    def test_forward3(self):
        x_data = np.arange(12).reshape((2, 2, 3))
        x = Variable(x_data)
        y = F.get_item(x, (Ellipsis, 2))
        self.assertTrue(np.allclose(y.data, x_data[..., 2]))

    def test_backward1(self):
        x_data = np.array([[1, 2, 3], [4, 5, 6]])
        slices = 1
        y_grad = np.ones(x_data[slices].shape)
        f = lambda x: F.get_item(x, slices)
        check_backward(f, x_data, y_grad=y_grad)

    def test_backward2(self):
        x_data = np.arange(12).reshape(4, 3)
        slices = slice(1, 3)
        y_grad = np.ones(x_data[slices].shape)
        f = lambda x: F.get_item(x, slices)
        check_backward(f, x_data, y_grad=y_grad)