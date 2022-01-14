from unittest.mock import Mock, patch

from tests import factories
from torchdemon.inference_queue import InferenceQueue
from torchdemon.models import InferencePayload


def test_process_request() -> None:
    inference_queue = InferenceQueue(batch_size=2, max_wait_ns=10)

    inference_payload1 = InferencePayload(
        client_id=factories.uuid_rand(), data=factories.inference_data_rand()
    )
    inference_payload2 = InferencePayload(
        client_id=factories.uuid_rand(), data=factories.inference_data_rand()
    )

    inference_payloads_batch = inference_queue.process_request(inference_payload1)
    assert inference_payloads_batch is None
    inference_payloads_batch = inference_queue.process_request(inference_payload2)
    assert inference_payloads_batch == [inference_payload1, inference_payload2]


@patch("torchdemon.inference_queue.time.time_ns")
def test_check_wait(mock_time_ns: Mock) -> None:
    mock_time_ns.side_effect = [
        0,  # First call at init
        5,  # Check now first time
        15,  # Check now second time
        15,  # Called at reset
    ]

    # First call to time_ns at init, _last_ns mocked to 0
    inference_queue = InferenceQueue(batch_size=2, max_wait_ns=10)
    inference_payload = InferencePayload(
        client_id=factories.uuid_rand(), data=factories.inference_data_rand()
    )
    inference_queue._payloads_batch = [inference_payload]

    # now mocked to 5, now - self._last_ns = 5 and 5 < 10
    # -> batch shouldn't be returned
    inference_payloads_batch = inference_queue.check_wait()
    assert inference_payloads_batch is None

    # now mocked to 15, now - self._last_ns = 10 and 10 <= 10
    # -> batch should be returned
    inference_payloads_batch = inference_queue.check_wait()
    assert inference_payloads_batch == [inference_payload]
