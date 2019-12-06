import unittest
import numpy as np
from dezero import Variable
import chainer
import dezero.functions as F
from dezero.utils import gradient_check


class TestLinear(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        w = Variable(x.data.T)
        b = None
        y = F.linear(x, w, b)

        res = y.data
        expected = np.array([[14, 32], [32, 77]])
        self.assertTrue(np.array_equal(res, expected))

    def test_forward2(self):
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype('f')
        W = x.T
        b = None
        y = F.linear(x, W, b)

        cy = chainer.functions.linear(x, W.T)
        self.assertTrue(np.array_equal(y.data, cy.data))

    def test_forward3(self):
        layer = chainer.links.Linear(3, 2)
        x = np.array([[1, 2, 3], [4, 5, 6]]).astype('f')
        W = layer.W.data.T
        b = layer.b.data
        y = F.linear(x, W, b)

        cy = layer(x)
        self.assertTrue(np.array_equal(y.data, cy.data))

    def test_backward1(self):
        x = np.random.randn(3, 2)
        W = np.random.randn(2, 3)
        b = np.random.randn(3)
        f = lambda x: F.linear(x, W, b)
        self.assertTrue(gradient_check(f, x))

    def test_backward1(self):
        x = np.random.randn(3, 2)
        W = np.random.randn(2, 3)
        b = np.random.randn(3)
        f = lambda x: F.linear(x, W, b)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        x = np.random.randn(100, 200)
        W = np.random.randn(200, 300)
        b = None
        f = lambda x: F.linear(x, W, b)
        self.assertTrue(gradient_check(f, x))