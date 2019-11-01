import unittest
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import check_backward
import chainer.functions as CF


class TestSoftmax(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y2 = CF.softmax(x, axis=1)
        y = F.softmax(Variable(x))
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward2(self):
        np.random.seed(0)
        x = np.random.rand(10, 10).astype('f')
        y2 = CF.softmax(x, axis=1)
        y = F.softmax(Variable(x))
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward3(self):
        np.random.seed(0)
        x = np.random.rand(10, 10, 10).astype('f')
        y2 = CF.softmax(x, axis=1)
        y = F.softmax(Variable(x))
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        f = lambda x: F.softmax(x, axis=1)
        self.assertTrue(check_backward(f, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        f = lambda x: F.softmax(x, axis=1)
        self.assertTrue(check_backward(f, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: F.softmax(x, axis=1)
        self.assertTrue(check_backward(f, x_data))


class TestSoftmaxCrossEntropy(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        y = F.softmax_cross_entropy(x, t)
        y2 = CF.softmax_cross_entropy(x, t)
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        f = lambda x: F.softmax_cross_entropy(x, Variable(t))
        self.assertTrue(check_backward(f, x_data))