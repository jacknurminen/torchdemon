from collections import defaultdict
from typing import Dict, List, Optional
from uuid import UUID

import numpy as np

from torchdemon.inference_model import InferenceModel
from torchdemon.inference_queue import InferenceQueue
from torchdemon.models import InferencePayload, InferenceResult


class InferenceScheduler:
    def __init__(
        self, inference_model: InferenceModel, inference_queue: InferenceQueue
    ):
        self._inference_model = inference_model
        self._inference_queue = inference_queue

    def check(
        self, inference_payload: InferencePayload
    ) -> Optional[List[InferenceResult]]:
        if inference_payloads_batch := self._inference_queue.check_batch_size(
            inference_payload=inference_payload
        ):
            return self._run_inference(inference_payloads_batch)

        if inference_payloads_batch := self._inference_queue.check_wait():
            return self._run_inference(inference_payloads_batch)
        return None

    def _run_inference(
        self, inference_payloads_batch: List[InferencePayload]
    ) -> List[InferenceResult]:
        client_ids, args, kwargs = [], [], []
        for payload in inference_payloads_batch:
            client_ids.append(payload.client_id)
            args.append(payload.data.args)
            kwargs.append(payload.data.kwargs)

        stacked_input_args = self._stack_arg_inputs(args)
        stacked_input_kwargs = self._stack_kwarg_inputs(kwargs)

        outputs = self._inference_model.infer(
            *stacked_input_args, **stacked_input_kwargs
        )

        unstacked_outputs = self._unstack_outputs(client_ids, outputs)
        return [
            InferenceResult(client_id=client_id, data=unstacked_outputs[client_id])
            for client_id in client_ids
        ]

    @staticmethod
    def _stack_arg_inputs(input_args_batch: List[List[np.ndarray]]) -> List[np.ndarray]:
        return [np.vstack(ndarrs) for ndarrs in zip(*input_args_batch)]

    @staticmethod
    def _stack_kwarg_inputs(
        input_kwargs_batch: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        batched_input_kwargs = defaultdict(list)
        for input_kwargs in input_kwargs_batch:
            for k, ndarr in input_kwargs.items():
                batched_input_kwargs[k].append(ndarr)
        return {k: np.vstack(ndarrs) for k, ndarrs in batched_input_kwargs.items()}

    @staticmethod
    def _unstack_outputs(
        row_ids: List[UUID], output: List[np.ndarray]
    ) -> Dict[UUID, List[np.ndarray]]:
        row_outputs = defaultdict(list)
        for i, row_id in enumerate(row_ids):
            for ndarr in output:
                row_outputs[row_id].append(ndarr[i])
        return row_outputs
