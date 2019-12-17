import unittest
import cupy as np  # !! CUPY !!
import dezero
import chainer
import chainer.functions as CF
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check, array_allclose


def get_params(N, C, H=None, W=None, dtype='f'):
    if H is not None:
        x = np.random.randn(N, C, H, W).astype(dtype)
    else:
        x = np.random.randn(N, C).astype(dtype)
    gamma = np.random.randn(C).astype(dtype)
    beta = np.random.randn(C).astype(dtype)
    mean = np.random.randn(C).astype(dtype)
    var = np.abs(np.random.randn(C).astype(dtype))
    return x, gamma, beta, mean, var


class TestFixedBatchNorm(unittest.TestCase):

    def test_type1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C)
        with dezero.test_mode():
            y = F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(y.data.dtype == np.float32)

    def test_forward1(self):
        N, C = 8, 1
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
        with dezero.test_mode():
            y = F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward2(self):
        N, C = 1, 10
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
        with dezero.test_mode():
            y = F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward3(self):
        N, C = 20, 10
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
        with dezero.test_mode():
            y = F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward4(self):
        N, C, H, W = 20, 10, 5, 5
        x, gamma, beta, mean, var = get_params(N, C, H, W)
        cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
        with dezero.test_mode():
            y = F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))




class TestBatchNorm(unittest.TestCase):

    def test_type1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C)
        y = F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(y.data.dtype == np.float32)

    def test_forward1(self):
        N, C = 8, 1
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.batch_normalization(x, gamma, beta, running_mean=mean, running_var=var)
        y = F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward2(self):
        N, C = 1, 10
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.batch_normalization(x, gamma, beta)
        y = F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward3(self):
        N, C = 20, 10
        x, gamma, beta, mean, var = get_params(N, C)
        cy = CF.batch_normalization(x, gamma, beta)
        y = F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward4(self):
        N, C, H, W = 20, 10, 5, 5
        x, gamma, beta, mean, var = get_params(N, C, H, W)
        cy = CF.batch_normalization(x, gamma, beta)
        y = F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_backward1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C, dtype=np.float64)
        f = lambda x: F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, x))

    def test_backward2(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C, dtype=np.float64)
        f = lambda gamma: F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, gamma))

    def test_backward3(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C, dtype=np.float64)
        f = lambda beta: F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, beta))

    def test_backward4(self):
        params = 10, 20, 5, 5
        x, gamma, beta, mean, var = get_params(*params, dtype=np.float64)
        f = lambda x: F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, x))

    def test_backward5(self):
        params = 10, 20, 5, 5
        x, gamma, beta, mean, var = get_params(*params, dtype=np.float64)
        f = lambda gamma: F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, gamma))

    def test_backward6(self):
        params = 10, 20, 5, 5
        x, gamma, beta, mean, var = get_params(*params, dtype=np.float64)
        f = lambda beta: F.batch_nrom(x, gamma, beta, mean, var)
        self.assertTrue(gradient_check(f, beta))


class TestBatchNormLayer(unittest.TestCase):

    def test_forward1(self):
        N, C = 8, 3
        x, gamma, beta, mean, var = get_params(N, C)
        cy = chainer.links.BatchNormalization(3)(x)
        y = dezero.layers.BatchNorm()(x)
        self.assertTrue(array_allclose(y.data, cy.data))

    def test_forward2(self):
        N, C = 8, 3
        cl = chainer.links.BatchNormalization(C)
        l = dezero.layers.BatchNorm()
        for i in range(10):
            x, gamma, beta, mean, var = get_params(N, C)
            cy = cl(x)
            y = l(x)
        self.assertTrue(array_allclose(cl.avg_mean, l.avg_mean.data))
        self.assertTrue(array_allclose(cl.avg_var, l.avg_var.data))

        with dezero.test_mode():
            y = l(x)
        with chainer.using_config('train', False):
            cy = cl(x)
        self.assertTrue(array_allclose(cy.data, y.data))