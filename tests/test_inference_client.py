from unittest.mock import Mock

import numpy as np

from tests import factories
from tests.utils import assert_ndarray_eq
from torchdemon import InferenceClient
from torchdemon.models import InferenceRequest, InferenceResult, Signal


def test_forward() -> None:
    mock_connection = Mock()
    mock_connection.poll.return_value = True
    inference_client = InferenceClient(mock_connection)

    mock_inference_result = InferenceResult(
        client_id=inference_client.client_id, data=[factories.ndarray_rand(5)]
    )
    mock_connection.recv.return_value = mock_inference_result

    forward_arg = factories.ndarray_rand(4)
    forward_kwarg = factories.ndarray_rand(6)
    inference_output = inference_client.forward(forward_arg, input=forward_kwarg)

    assert mock_connection.poll.call_count == 1
    assert mock_connection.send.call_count == 1
    assert mock_connection.recv.call_count == 1
    assert isinstance(inference_output, np.ndarray)
    assert_ndarray_eq(inference_output, mock_inference_result.data[0])

    sent_args, _ = mock_connection.send.call_args_list[0]
    sent_inference_request = sent_args[0]
    assert_ndarray_eq(forward_arg, sent_inference_request.data.args[0])
    assert_ndarray_eq(forward_kwarg, sent_inference_request.data.kwargs["input"])


def test_close() -> None:
    mock_connection = Mock()
    inference_client = InferenceClient(mock_connection)

    inference_client.close()

    mock_connection.send.assert_called_once_with(
        InferenceRequest(client_id=inference_client.client_id, data=Signal.CLOSE)
    )
    mock_connection.close.assert_called_once()
