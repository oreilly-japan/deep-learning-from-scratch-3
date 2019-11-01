import unittest
import numpy as np
import dezero.layers as L
import dezero.functions as F
from dezero.utils import check_backward
import chainer.functions as CF


class TestPooling_simple(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.pooling_simple(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.pooling_simple(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad, cover_all=False)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')
        f = lambda x: F.pooling_simple(x, ksize, stride, pad)
        self.assertTrue(check_backward(f, x))


class TestPooling(unittest.TestCase):

    def test_forward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.pooling(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_forward2(self):
        n, c, h, w = 1, 5, 15, 15
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f')

        y = F.pooling(x, ksize, stride, pad)
        expected = CF.max_pooling_2d(x, ksize, stride, pad, cover_all=False)
        self.assertTrue(np.array_equal(expected.data, y.data))

    def test_backward1(self):
        n, c, h, w = 1, 5, 16, 16
        ksize, stride, pad = 2, 2, 0
        x = np.random.randn(n, c, h, w).astype('f') * 1000
        f = lambda x: F.pooling(x, ksize, stride, pad)
        self.assertTrue(check_backward(f, x))