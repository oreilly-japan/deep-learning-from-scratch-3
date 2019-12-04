import unittest
import numpy as np
import chainer
import dezero
from dezero import Variable
import dezero.functions as F
from dezero.utils import check_backward
from dezero.models import VGG16


class TestVGG16(unittest.TestCase):
    def test_forward1(self):
        x = np.random.randn(1, 3, 224, 224).astype('f')
        _model = chainer.links.VGG16Layers(None)

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                out_layer_name = 'fc8'
                _y = _model.forward(x, [out_layer_name])[out_layer_name]

        model = VGG16()
        layers = _model.available_layers
        for l in layers:
            if "conv" in l or "fc" in l:
                m1 = getattr(model, l)
                m2 = getattr(_model, l)
                m1.W.data = m2.W.data
                m1.b.data = m2.b.data
                if "fc" in l:
                    m1.W.data = m1.W.data.T

        with dezero.test_mode():
            y = model(x)
        res = np.array_equal(y.data, _y.data)
        self.assertTrue(res)


    """
    def test_backward1(self):
        x = np.random.randn(1, 3, 224, 224)
        model = VGG16()
        def f(x):
            with dezero.test_mode():
                return model(x)
        self.assertTrue(check_backward(f, x))
    """