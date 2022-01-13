import uuid

import numpy as np

from torchdemon.models import INFERENCE_DATA_T, InferencePayload
from torchdemon.queue import InferenceQueue


def test_process_request() -> None:
    inference_queue = InferenceQueue(batch_size=2, max_wait_ns=10)

    inference_payload1 = InferencePayload(
        client_id=_dummy_uuid(), data=_dummy_inference_data()
    )
    inference_payload2 = InferencePayload(
        client_id=_dummy_uuid(), data=_dummy_inference_data()
    )

    inference_payloads_batch = inference_queue.process_request(inference_payload1)
    assert inference_payloads_batch is None
    inference_payloads_batch = inference_queue.process_request(inference_payload2)
    assert inference_payloads_batch == [inference_payload1, inference_payload2]


def _dummy_uuid() -> "uuid.UUID":
    return uuid.uuid4()


def _dummy_inference_data() -> INFERENCE_DATA_T:
    return {"input": _dummy_ndarray()}


def _dummy_ndarray() -> "np.ndarray":
    return np.array([1.0, 2.0])
