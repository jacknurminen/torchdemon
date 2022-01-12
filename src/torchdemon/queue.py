import time
from typing import List, Optional

from torchdemon.models import InferencePayload


class InferenceQueue:
    def __init__(self, batch_size: int, max_wait_ns: int):
        self._batch_size = batch_size
        self._max_wait_ns = max_wait_ns
        self._payloads_batch: List[InferencePayload] = []
        self._last_ns = time.time_ns()

    def process_request(
        self, inference_payload: InferencePayload
    ) -> Optional[List[InferencePayload]]:
        self._payloads_batch.append(inference_payload)
        if len(self._payloads_batch) >= self._batch_size:
            result = self._payloads_batch
            self._reset()
            return result
        return None

    def check_wait(self) -> Optional[List[InferencePayload]]:
        now = time.time_ns()
        if self._payloads_batch and now - self._last_ns >= self._max_wait_ns:
            result = self._payloads_batch
            self._reset()
            return result
        return None

    def _reset(self) -> None:
        self._payloads_batch = []
        self._last_ns = time.time_ns()
