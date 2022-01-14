from unittest.mock import Mock, call, patch

import pytest

from tests import factories
from tests.utils import assert_ndarray_eq
from torchdemon.inference_scheduler import InferenceScheduler
from torchdemon.models import InferencePayload, InferenceRequest, Signal


@pytest.fixture
def mock_connection() -> Mock:
    with patch("torchdemon.inference_scheduler.Connection") as mock_connection:
        return mock_connection


@pytest.fixture
def mock_inference_model() -> Mock:
    with patch("torchdemon.inference_scheduler.InferenceModel") as mock_inference_model:
        return mock_inference_model


@pytest.fixture
def mock_inference_queue() -> Mock:
    with patch("torchdemon.inference_scheduler.InferenceQueue") as mock_inference_queue:
        return mock_inference_queue


def test_create_client(mock_inference_model: Mock, mock_inference_queue: Mock) -> None:
    inference_scheduler = InferenceScheduler(mock_inference_model, mock_inference_queue)

    inference_client = inference_scheduler.create_client()
    assert inference_client.client_id in inference_scheduler._connections


def test_check(
    mock_inference_model: Mock, mock_inference_queue: Mock, mock_connection: Mock
) -> None:
    client_id = factories.uuid_const()
    inference_data = factories.inference_data_rand()
    inference_request = InferenceRequest(client_id=client_id, data=inference_data)
    inference_payload = InferencePayload(client_id=client_id, data=inference_data)

    mock_connection.recv.return_value = inference_request
    mock_inference_queue.check_wait.return_value = None
    mock_inference_queue.process_request.return_value = [
        InferencePayload(client_id=client_id, data=inference_data),
        InferencePayload(client_id=client_id, data=inference_data),
    ]
    infer_output = factories.ndarray_rand(2, 5)
    mock_inference_model.infer.return_value = {"output": infer_output}

    inference_scheduler = InferenceScheduler(mock_inference_model, mock_inference_queue)
    inference_scheduler._connections[client_id] = mock_connection

    for poll_return_value in [False, True]:
        mock_connection.poll.return_value = poll_return_value
        inference_scheduler.check()

    assert mock_connection.poll.call_count == 2
    assert mock_connection.recv.call_count == 1
    mock_connection.close.assert_not_called()
    assert mock_inference_queue.process_request.call_args == call(
        inference_payload=inference_payload
    )
    assert mock_inference_model.infer.call_count == 1
    assert mock_connection.send.call_count == 2

    send_args, _ = mock_connection.send.call_args_list[0]
    assert_ndarray_eq(send_args[0]["output"], infer_output[0])
    send_args, _ = mock_connection.send.call_args_list[1]
    assert_ndarray_eq(send_args[0]["output"], infer_output[1])


def test_close_connection(
    mock_inference_model: Mock, mock_inference_queue: Mock, mock_connection: Mock
) -> None:
    client_id = factories.uuid_const()
    inference_request = InferenceRequest(client_id=client_id, data=Signal.CLOSE)

    mock_connection.recv.return_value = inference_request
    mock_inference_queue.check_wait.return_value = None

    inference_scheduler = InferenceScheduler(mock_inference_model, mock_inference_queue)
    inference_scheduler._connections[client_id] = mock_connection

    mock_connection.poll.return_value = True
    assert inference_scheduler.connections_open()

    inference_scheduler.check()

    assert mock_connection.poll.call_count == 1
    assert mock_connection.recv.call_count == 1
    assert mock_connection.close.call_count == 1
    assert not inference_scheduler.connections_open()
    assert not inference_scheduler._connections
