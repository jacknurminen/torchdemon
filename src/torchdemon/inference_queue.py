import time
from typing import List, Optional

from torchdemon.models import InferencePayload


class InferenceQueue:
    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_wait_ns: Optional[int] = None,
    ):
        self._batch_size = batch_size
        self._max_wait_ns = max_wait_ns
        self._payloads_batch: List[InferencePayload] = []
        self._last_ns = time.time_ns()

    def check_batch_size(
        self, inference_payload: InferencePayload
    ) -> Optional[List[InferencePayload]]:
        self._payloads_batch.append(inference_payload)
        if self._batch_size and len(self._payloads_batch) >= self._batch_size:
            result = self._payloads_batch
            self._reset()
            return result
        return None

    def check_wait(self) -> Optional[List[InferencePayload]]:
        if self._max_wait_ns:
            now = time.time_ns()
            if self._payloads_batch and now - self._last_ns >= self._max_wait_ns:
                result = self._payloads_batch
                self._reset()
                return result
        return None

    def _reset(self) -> None:
        self._payloads_batch = []
        self._last_ns = time.time_ns()
