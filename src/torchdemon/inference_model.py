from typing import Generic, List

import numpy as np
import torch

from torchdemon.models import TORCH_MODEL_T


class InferenceModel(Generic[TORCH_MODEL_T]):
    def __init__(self, model: TORCH_MODEL_T, device: torch.device):
        self._model = model
        self._device = device
        self._model.to(device)

    def infer(self, *args: np.ndarray, **kwargs: np.ndarray) -> List[np.ndarray]:
        arg_tensors = [self._to_tensor(arg) for arg in args]
        kwarg_tensors = {kw: self._to_tensor(arg) for kw, arg in kwargs.items()}
        with torch.no_grad():
            outputs = self._model.forward(*arg_tensors, **kwarg_tensors)
        if isinstance(outputs, tuple):
            return [self._to_ndarray(output) for output in outputs]
        return [self._to_ndarray(outputs)]

    def _to_tensor(self, ndarr: np.ndarray) -> torch.Tensor:
        if ndarr.dtype == np.float64:
            return torch.from_numpy(ndarr).to(torch.float64).to(self._device)
        if ndarr.dtype == np.float32:
            return torch.from_numpy(ndarr).to(torch.float32).to(self._device)
        if ndarr.dtype == np.int64:
            return torch.from_numpy(ndarr).to(torch.int64).to(self._device)
        if ndarr.dtype == np.int32:
            return torch.from_numpy(ndarr).to(torch.int32).to(self._device)
        raise AssertionError(f"Unsupported dtype {ndarr.dtype}")

    @staticmethod
    def _to_ndarray(tensor: torch.Tensor) -> np.ndarray:
        ndarr: np.ndarray = tensor.cpu().numpy()
        return ndarr
