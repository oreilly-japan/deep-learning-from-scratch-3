import unittest
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import check_backward


class TestTranspose(unittest.TestCase):
    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = F.transpose(x)
        self.assertEqual(y.shape, (3, 2))

    def test_backward1(self):
        x_data = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(check_backward(F.transpose, x_data))

    def test_backward2(self):
        x_data = np.array([1, 2, 3])
        self.assertTrue(check_backward(F.transpose, x_data))

    def test_backward3(self):
        x_data = np.random.randn(10, 5)
        self.assertTrue(check_backward(F.transpose, x_data))