import unittest
import cupy as np  # !! CUPY !!
from dezero import Variable
import dezero.functions as F
from dezero.utils import gradient_check, array_allclose, array_equal


class TestRelu(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        res = F.relu(x)
        ans = np.array([[0, 0], [2, 0], [0, 1]], np.float32)
        self.assertTrue(array_allclose(res, ans))

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])
        self.assertTrue(gradient_check(F.relu, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10) * 100
        self.assertTrue(gradient_check(F.relu, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10) * 100
        self.assertTrue(gradient_check(F.relu, x_data))


class TestLeakyRelu(unittest.TestCase):

    def test_forward1(self):
        x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        res = F.leaky_relu(x)
        ans = np.array([[-0.2, 0.], [2., -0.6], [-0.4, 1.]], np.float32)
        self.assertTrue(array_allclose(res, ans))

    def test_backward1(self):
        x_data = np.array([[-1, 1, 2], [-1, 2, 4]])
        self.assertTrue(gradient_check(F.leaky_relu, x_data))

    def test_backward2(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10) * 100
        self.assertTrue(gradient_check(F.leaky_relu, x_data))

    def test_backward3(self):
        np.random.seed(0)
        x_data = np.random.rand(10, 10, 10) * 100
        self.assertTrue(gradient_check(F.leaky_relu, x_data))