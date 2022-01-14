from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Union
from uuid import UUID

import numpy as np


class Signal(IntEnum):
    CLOSE = 0


INFERENCE_DATA_T = Dict[str, np.ndarray]

INFERENCE_REQUEST_DATA_T = Union[INFERENCE_DATA_T, Signal]


@dataclass
class InferenceRequest:
    client_id: UUID
    data: INFERENCE_REQUEST_DATA_T


@dataclass
class InferencePayload:
    client_id: UUID
    data: INFERENCE_DATA_T


@dataclass
class InferenceResult:
    client_id: UUID
    data: INFERENCE_DATA_T
