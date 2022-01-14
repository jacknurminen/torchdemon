import numpy as np


def assert_ndarray_eq(ndarr1: np.ndarray, ndarr2: np.ndarray) -> None:
    assert ndarr1.shape == ndarr2.shape
    assert np.sum(ndarr1 - ndarr2) == 0
