import unittest
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check
import chainer.functions as CF


class TestSoftmaxCrossEntropy(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        y = F.softmax_cross_entropy(x, t)
        y2 = CF.softmax_cross_entropy(x, t)
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        f = lambda x: F.softmax_cross_entropy(x, Variable(t))
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        N, CLS_NUM = 10, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: F.softmax_cross_entropy(x, t)
        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        N, CLS_NUM = 100, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: F.softmax_cross_entropy(x, t)
        self.assertTrue(gradient_check(f, x))


class TestSoftmaxCrossEntropy_simple(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        y = F.softmax_cross_entropy_simple(x, t)
        y2 = CF.softmax_cross_entropy(x, t)
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        t = np.array([3, 0]).astype(np.int32)
        f = lambda x: F.softmax_cross_entropy_simple(x, Variable(t))
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        N, CLS_NUM = 10, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: F.softmax_cross_entropy_simple(x, t)
        self.assertTrue(gradient_check(f, x))

    def test_backward3(self):
        N, CLS_NUM = 100, 10
        x = np.random.randn(N, CLS_NUM)
        t = np.random.randint(0, CLS_NUM, (N,))
        f = lambda x: F.softmax_cross_entropy_simple(x, t)
        self.assertTrue(gradient_check(f, x))