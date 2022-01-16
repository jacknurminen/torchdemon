from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, TypeVar, Union
from uuid import UUID

if TYPE_CHECKING:
    import numpy as np
    import torch

TORCH_MODEL_T = TypeVar("TORCH_MODEL_T", bound="torch.nn.Module")


class Signal(IntEnum):
    CLOSE = 0


@dataclass
class InferenceInputData:
    args: List["np.ndarray"]
    kwargs: Dict[str, "np.ndarray"]


@dataclass
class InferenceRequest:
    client_id: UUID
    data: Union[InferenceInputData, Signal]


@dataclass
class InferencePayload:
    client_id: UUID
    data: InferenceInputData


@dataclass
class InferenceResult:
    client_id: UUID
    data: List["np.ndarray"]
