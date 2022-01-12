import copy
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from torchdemon.models import INFERENCE_DATA_T

if TYPE_CHECKING:
    import numpy as np
    import torch

ModelT = TypeVar("ModelT", bound="torch.nn.Module")


class InferenceModel(Generic[ModelT]):
    def __init__(self, device: "torch.device"):
        self.device = device
        self.model: Optional["torch.nn.Module"] = None

    def load_model(self, model: ModelT) -> None:
        self.model = torch.jit.script(copy.deepcopy(model))
        assert self.model
        self.model.to(self.device)
        self.model.eval()

    def infer(self, **inputs: np.ndarray) -> INFERENCE_DATA_T:
        raise NotImplementedError
