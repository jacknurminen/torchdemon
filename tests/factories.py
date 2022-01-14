from uuid import UUID, uuid4

import numpy as np

from torchdemon.models import (
    INFERENCE_DATA_T,
    InferencePayload,
    InferenceRequest,
    InferenceResult,
)


def uuid_const() -> UUID:
    return UUID("009b7240-c7e5-4df3-8722-c1be32390106")


def uuid_rand() -> UUID:
    return uuid4()


def ndarray() -> np.ndarray:
    return np.arange(start=0.0, stop=10.0, step=2.0)


def ndarray_rand(*args: int) -> np.ndarray:
    return np.random.rand(*args)


def inference_data_const() -> INFERENCE_DATA_T:
    return {"input": ndarray()}


def inference_data_rand() -> INFERENCE_DATA_T:
    return {"input": ndarray_rand(5)}


def inference_request() -> InferenceRequest:
    return InferenceRequest(client_id=uuid_const(), data=inference_data_rand())


def inference_payload() -> InferencePayload:
    return InferencePayload(client_id=uuid_const(), data=inference_data_rand())


def inference_result() -> InferenceResult:
    return InferenceResult(client_id=uuid_const(), data=inference_data_rand())
