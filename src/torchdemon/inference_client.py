import uuid
from typing import TYPE_CHECKING

from torchdemon.models import InferenceRequest, InferenceResult, Signal

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    import numpy as np


class InferenceClient:
    def __init__(self, connection: Connection):
        self._connection = connection
        self.client_id = uuid.uuid4()

    def forward(self, **inputs: "np.ndarray") -> InferenceResult:
        self._connection.send(InferenceRequest(self.client_id, data=inputs))
        while True:
            if self._connection.poll():
                inference_result: InferenceResult = self._connection.recv()
                return inference_result

    def close(self) -> None:
        self._connection.send(InferenceRequest(self.client_id, data=Signal.KILL))
        self._connection.close()
