from unittest.mock import Mock, patch

import pytest

from tests import factories
from tests.utils import assert_ndarray_eq
from torchdemon.inference_client import InferenceClient
from torchdemon.models import InferenceRequest, Signal


@pytest.fixture
def mock_connection() -> Mock:
    with patch("torchdemon.inference_client.Connection") as mock_connection:
        mock_connection.poll.return_value = True
        return mock_connection


def test_forward(mock_connection: Mock) -> None:
    inference_client = InferenceClient(mock_connection)

    mock_inference_result = factories.inference_result()
    mock_inference_result.client_id = inference_client.client_id
    mock_connection.recv.return_value = mock_inference_result

    inference_request = factories.inference_payload()
    inference_request.client_id = inference_client.client_id

    inference_result = inference_client.forward(input=inference_request.data["input"])

    send_args, _ = mock_connection.send.call_args_list[0]
    send_args[0].client_id = inference_request.client_id
    assert_ndarray_eq(send_args[0].data["input"], inference_request.data["input"])

    mock_connection.poll.assert_called_once()
    assert inference_result == mock_inference_result


def test_close(mock_connection: Mock) -> None:
    inference_client = InferenceClient(mock_connection)
    inference_request = InferenceRequest(
        client_id=inference_client.client_id, data=Signal.CLOSE
    )

    inference_client.close()

    mock_connection.send.assert_called_once_with(inference_request)
    mock_connection.close.assert_called_once()
