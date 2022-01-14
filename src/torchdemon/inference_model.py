import copy
from typing import Generic, Optional, TypeVar

import numpy as np
import torch

from torchdemon.models import INFERENCE_DATA_T

_TORCH_MODEL_T = TypeVar("_TORCH_MODEL_T", bound=torch.nn.Module)


class InferenceModel(Generic[_TORCH_MODEL_T]):
    def __init__(self, device: torch.device):
        self.device = device
        self.model: Optional[_TORCH_MODEL_T] = None

    def load_model(self, model: _TORCH_MODEL_T) -> None:
        self.model = torch.jit.script(copy.deepcopy(model))
        assert self.model
        self.model.to(self.device)
        self.model.eval()

    def infer(self, **inputs: np.ndarray) -> INFERENCE_DATA_T:
        raise NotImplementedError
