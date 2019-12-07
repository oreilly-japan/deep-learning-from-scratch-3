import unittest
import cupy as np  # !! CUPY !!
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check, array_allclose


class TestSum(unittest.TestCase):

    def test_datatype(self):
        x = Variable(np.random.rand(10))
        y = F.sum(x)
        # np.float64ではなく0次元のnp.ndarrayを返す
        self.assertFalse(np.isscalar(y))

    def test_forward1(self):
        x = Variable(np.array(2.0))
        y = F.sum(x)
        expected = np.sum(x.data)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward2(self):
        x = Variable(np.random.rand(10, 20, 30))
        y = F.sum(x, axis=1)
        expected = np.sum(x.data, axis=1)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward3(self):
        x = Variable(np.random.rand(10, 20, 30))
        y = F.sum(x, axis=1, keepdims=True)
        expected = np.sum(x.data, axis=1, keepdims=True)
        self.assertTrue(array_allclose(y.data, expected))

    def test_backward1(self):
        x_data = np.random.rand(10)
        f = lambda x: F.sum(x)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.rand(10, 10)
        f = lambda x: F.sum(x, axis=1)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 20, 20)
        f = lambda x: F.sum(x, axis=2)
        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.rand(10, 20, 20)
        f = lambda x: F.sum(x, axis=None)
        self.assertTrue(gradient_check(f, x_data))


class TestSumTo(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.random.rand(10))
        y = F.sum_to(x, (1,))
        expected = np.sum(x.data)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward2(self):
        x = Variable(np.array([[1., 2., 3.], [4., 5., 6.]]))
        y = F.sum_to(x, (1, 3))
        expected = np.sum(x.data, axis=0, keepdims=True)
        self.assertTrue(array_allclose(y.data, expected))

    def test_forward3(self):
        x = Variable(np.random.rand(10))
        y = F.sum_to(x, (10,))
        expected = x.data  # 同じ形状なので何もしない
        self.assertTrue(array_allclose(y.data, expected))

    def test_backward1(self):
        x_data = np.random.rand(10)
        f = lambda x: F.sum_to(x, (1,))
        self.assertTrue(gradient_check(f, x_data))

    def test_backward2(self):
        x_data = np.random.rand(10, 10) * 10
        f = lambda x: F.sum_to(x, (10,))
        self.assertTrue(gradient_check(f, x_data))

    def test_backward3(self):
        x_data = np.random.rand(10, 20, 20) * 100
        f = lambda x: F.sum_to(x, (10,))
        self.assertTrue(gradient_check(f, x_data))

    def test_backward4(self):
        x_data = np.random.rand(10)
        f = lambda x: F.sum_to(x, (10,)) + 1
        self.assertTrue(gradient_check(f, x_data))