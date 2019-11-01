import numpy as np
gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False


def get_array_module(array):
    if not gpu_enable:
        return np
    xp = cp.get_array_module(array)
    return xp


def as_numpy(array):
    if isinstance(array, np.ndarray):
        return array
    return cp.asnumpy(array)


def as_cupy(array):
    if not gpu_enable:
        msg = "DeZero's GPU mode requires CuPy."
        raise Exception(msg)
    return cp.asarray(array)