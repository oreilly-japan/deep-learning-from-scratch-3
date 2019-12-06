import unittest
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check
import chainer.functions as CF


class TestSigmoid(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y2 = CF.sigmoid(x)
        y = F.sigmoid(Variable(x))
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward2(self):
        x = np.random.randn(10, 10).astype(np.float32)
        y2 = CF.sigmoid(x)
        y = F.sigmoid(Variable(x))
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        self.assertTrue(gradient_check(F.sigmoid, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        self.assertTrue(gradient_check(F.sigmoid, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        self.assertTrue(gradient_check(F.sigmoid, x_data))