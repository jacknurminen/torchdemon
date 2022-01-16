from uuid import UUID, uuid4

import numpy as np

from torchdemon.models import InferenceInputData


def uuid_const() -> UUID:
    return UUID("009b7240-c7e5-4df3-8722-c1be32390106")


def uuid_rand() -> UUID:
    return uuid4()


def ndarray_randint(*args: int) -> np.ndarray:
    return np.random.randint(0, 10, size=args)


def ndarray_rand(*args: int) -> np.ndarray:
    return np.random.rand(*args)


def inference_input_data() -> InferenceInputData:
    return InferenceInputData(
        args=[ndarray_randint(2), ndarray_rand(4)], kwargs={"input": ndarray_rand(6)}
    )
