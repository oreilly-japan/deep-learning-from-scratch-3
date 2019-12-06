import unittest
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check
import chainer.functions as CF


class TestSoftmaxSimple(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[0, 1, 2], [0, 2, 4]], np.float32)
        y2 = CF.softmax(x, axis=1)
        y = F.softmax_simple(Variable(x))
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward2(self):
        np.random.seed(0)
        x = np.random.rand(10, 10).astype('f')
        y2 = CF.softmax(x, axis=1)
        y = F.softmax_simple(Variable(x))
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_forward3(self):
        np.random.seed(0)
        x = np.random.rand(10, 10, 10).astype('f')
        y2 = CF.softmax(x, axis=1)
        y = F.softmax_simple(Variable(x))
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x_data = np.array([[0, 1, 2], [0, 2, 4]])
        f = lambda x: F.softmax_simple(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        f = lambda x: F.softmax_simple(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: F.softmax_simple(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))


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
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        f = lambda x: F.softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: F.softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))


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
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10)
        f = lambda x: F.softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10)
        f = lambda x: F.softmax(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))


class TestLogSoftmax(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
        y = F.log_softmax(x)
        y2 = CF.log_softmax(x)
        res = np.allclose(y.data, y2.data)
        self.assertTrue(res)

    def test_backward1(self):
        x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]])
        f = lambda x: F.log_softmax(x)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(10, 10)
        f = lambda x: F.log_softmax(x)
        self.assertTrue(gradient_check(f, x))