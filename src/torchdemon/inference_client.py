import uuid
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

from torchdemon.models import InferenceRequest, InferenceResult, Signal

if TYPE_CHECKING:
    import numpy as np


class InferenceClient:
    def __init__(self, connection: Connection):
        self._connection = connection
        self.client_id = uuid.uuid4()

    def forward(self, **inputs: "np.ndarray") -> InferenceResult:
        inference_request = InferenceRequest(self.client_id, data=inputs)
        self._connection.send(inference_request)
        while True:
            if self._connection.poll():
                inference_result: InferenceResult = self._connection.recv()
                return inference_result

    def close(self) -> None:
        inference_request = InferenceRequest(self.client_id, data=Signal.CLOSE)
        self._connection.send(inference_request)
        self._connection.close()
