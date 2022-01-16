import uuid
from typing import TYPE_CHECKING

from torchdemon.models import (
    InferenceInputData,
    InferenceRequest,
    InferenceResult,
    Signal,
)

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    import numpy as np


class InferenceClient:
    def __init__(self, connection: "Connection"):
        self._connection = connection
        self.client_id = uuid.uuid4()

    def forward(self, *args: "np.ndarray", **kwargs: "np.ndarray") -> InferenceResult:
        inference_request = InferenceRequest(
            self.client_id, data=InferenceInputData(args=list(args), kwargs=kwargs)
        )
        self._connection.send(inference_request)
        while True:
            if self._connection.poll():
                inference_result: InferenceResult = self._connection.recv()
                return inference_result

    def close(self) -> None:
        inference_request = InferenceRequest(self.client_id, data=Signal.CLOSE)
        self._connection.send(inference_request)
        self._connection.close()
