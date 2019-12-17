import unittest
import cupy as np  # !! CUPY !!
import dezero.functions as F
from dezero.utils import gradient_check, array_allclose
import chainer.functions as CF


class TestPooling_simple(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.pooling_simple(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.pooling_simple(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad, cover_all=False)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f') * 100
        f = lambda x: F.pooling_simple(x, ksize, stride, pad)
        self.assertTrue(gradient_check(f, x))


class TestPooling(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.pooling(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.pooling(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad, cover_all=False)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f') * 1000
        f = lambda x: F.pooling(x, ksize, stride, pad)
        self.assertTrue(gradient_check(f, x))


class TestAveragePooling(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.average_pooling(x, ksize, stride, pad)
        expected = CF.average_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.average_pooling(x, ksize, stride, pad)
        expected = CF.average_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(array_allclose(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f') * 1000
        f = lambda x: F.average_pooling(x, ksize, stride, pad)
        self.assertTrue(gradient_check(f, x))