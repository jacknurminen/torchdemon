from unittest.mock import Mock, patch

import pytest
import torch

from tests import factories
from tests.utils import assert_ndarray_eq
from torchdemon.inference_server import InferenceServer
from torchdemon.models import InferenceRequest, Signal


@pytest.fixture
def mock_connection() -> Mock:
    with patch("torchdemon.inference_server.Connection") as mock:
        return mock


def mock_client() -> Mock:
    client = Mock()
    client.client_id = factories.uuid_rand()
    client.connection = Mock()
    return client


def test_create_client() -> None:
    inference_server = InferenceServer(Mock(), batch_size=2, max_wait_ns=10)
    inference_client = inference_server.create_client()
    assert inference_client.client_id in inference_server._connections


@patch("torchdemon.inference_queue.time.time_ns")
def test_check_batch_full(mock_time: Mock) -> None:
    mock_time.return_value = 0
    mock_torch_model = Mock()
    inference_server = InferenceServer(
        model=mock_torch_model, batch_size=2, max_wait_ns=10
    )
    client1 = mock_client()
    client2 = mock_client()

    inference_server._connections[client1.client_id] = client1.connection
    inference_server._connections[client2.client_id] = client2.connection

    torch_model_output = torch.rand(2, 4), torch.rand(2, 6)
    mock_torch_model.forward.return_value = torch_model_output

    # First client calls
    client1.connection.poll.return_value = True
    client1.connection.recv.return_value = InferenceRequest(
        client_id=client1.client_id, data=factories.inference_input_data()
    )
    client2.connection.poll.return_value = False

    inference_server.check()

    assert client1.connection.close.call_count == 0
    assert client1.connection.poll.call_count == 1
    assert client1.connection.recv.call_count == 1
    assert client1.connection.send.call_count == 0

    assert client2.connection.close.call_count == 0
    assert client2.connection.poll.call_count == 1
    assert client2.connection.recv.call_count == 0
    assert client2.connection.send.call_count == 0

    # Second client calls
    client1.connection.poll.return_value = False
    client2.connection.poll.return_value = True
    client2.connection.recv.return_value = InferenceRequest(
        client_id=client2.client_id, data=factories.inference_input_data()
    )

    # Batch full, model outputs returned
    inference_server.check()

    assert client1.connection.close.call_count == 0
    assert client1.connection.poll.call_count == 2
    assert client1.connection.recv.call_count == 1
    assert client1.connection.send.call_count == 1

    assert client2.connection.close.call_count == 0
    assert client2.connection.poll.call_count == 2
    assert client2.connection.recv.call_count == 1
    assert client2.connection.send.call_count == 1

    args, _ = client1.connection.send.call_args_list[0]
    inference_result = args[0]
    assert len(inference_result.data) == 2
    assert_ndarray_eq(inference_result.data[0], torch_model_output[0][0].numpy())
    assert_ndarray_eq(inference_result.data[1], torch_model_output[1][0].numpy())

    args, _ = client2.connection.send.call_args_list[0]
    inference_result = args[0]
    assert len(inference_result.data) == 2
    assert_ndarray_eq(inference_result.data[0], torch_model_output[0][1].numpy())
    assert_ndarray_eq(inference_result.data[1], torch_model_output[1][1].numpy())

    # First client closes connection
    client1.connection.poll.return_value = True
    client1.connection.recv.return_value = InferenceRequest(
        client_id=client1.client_id, data=Signal.CLOSE
    )
    client2.connection.poll.return_value = False

    # Batch full, model outputs returned
    inference_server.check()

    assert client1.connection.close.call_count == 1
    assert client1.connection.poll.call_count == 3
    assert client1.connection.recv.call_count == 2
    assert client1.connection.send.call_count == 1
    assert client1.client_id not in inference_server._connections

    assert client2.connection.close.call_count == 0
    assert client2.connection.poll.call_count == 3
    assert client2.connection.recv.call_count == 1
    assert client2.connection.send.call_count == 1
    assert client2.client_id in inference_server._connections

    # Second client closes connection
    client2.connection.poll.return_value = True
    client2.connection.recv.return_value = InferenceRequest(
        client_id=client2.client_id, data=Signal.CLOSE
    )

    # Batch full, model outputs returned
    inference_server.check()

    assert client1.connection.close.call_count == 1
    assert client1.connection.poll.call_count == 3
    assert client1.connection.recv.call_count == 2
    assert client1.connection.send.call_count == 1
    assert client1.client_id not in inference_server._connections

    assert client2.connection.close.call_count == 1
    assert client2.connection.poll.call_count == 4
    assert client2.connection.recv.call_count == 2
    assert client2.connection.send.call_count == 1
    assert client2.client_id not in inference_server._connections


@patch("torchdemon.inference_queue.time.time_ns")
def test_check_wait(mock_time: Mock) -> None:
    mock_time.return_value = 0
    mock_torch_model = Mock()
    inference_server = InferenceServer(
        model=mock_torch_model, batch_size=10, max_wait_ns=10
    )
    client1 = mock_client()
    client2 = mock_client()

    inference_server._connections[client1.client_id] = client1.connection
    inference_server._connections[client2.client_id] = client2.connection

    torch_model_output = torch.rand(2, 4)
    mock_torch_model.forward.return_value = torch_model_output

    # First client calls, before max wait
    mock_time.return_value = 5
    client1.connection.poll.return_value = True
    client1.connection.recv.return_value = InferenceRequest(
        client_id=client1.client_id, data=factories.inference_input_data()
    )
    client2.connection.poll.return_value = False

    inference_server.check()

    assert client1.connection.close.call_count == 0
    assert client1.connection.poll.call_count == 1
    assert client1.connection.recv.call_count == 1
    assert client1.connection.send.call_count == 0

    assert client2.connection.close.call_count == 0
    assert client2.connection.poll.call_count == 1
    assert client2.connection.recv.call_count == 0
    assert client2.connection.send.call_count == 0

    # Second client calls, at max wait time
    mock_time.return_value = 10
    client1.connection.poll.return_value = False
    client2.connection.poll.return_value = True
    client2.connection.recv.return_value = InferenceRequest(
        client_id=client2.client_id, data=factories.inference_input_data()
    )

    # Max wait reached, model outputs returned
    inference_server.check()

    assert client1.connection.close.call_count == 0
    assert client1.connection.poll.call_count == 2
    assert client1.connection.recv.call_count == 1
    assert client1.connection.send.call_count == 1

    assert client2.connection.close.call_count == 0
    assert client2.connection.poll.call_count == 2
    assert client2.connection.recv.call_count == 1
    assert client2.connection.send.call_count == 1

    args, _ = client1.connection.send.call_args_list[0]
    inference_result = args[0]
    assert len(inference_result.data) == 1
    assert_ndarray_eq(inference_result.data[0], torch_model_output[0].numpy())

    args, _ = client2.connection.send.call_args_list[0]
    inference_result = args[0]
    assert len(inference_result.data) == 1
    assert_ndarray_eq(inference_result.data[0], torch_model_output[1].numpy())


@patch("torchdemon.inference_queue.time.time_ns")
def test_connections_open(mock_time: Mock) -> None:
    mock_time.return_value = 0
    mock_torch_model = Mock()
    inference_server = InferenceServer(
        model=mock_torch_model, batch_size=10, max_wait_ns=10
    )
    client1 = mock_client()

    inference_server._connections[client1.client_id] = client1.connection

    assert inference_server.connections_open()

    client1.connection.poll.return_value = True
    client1.connection.recv.return_value = InferenceRequest(
        client_id=client1.client_id, data=Signal.CLOSE
    )

    inference_server.check()

    assert not inference_server.connections_open()
