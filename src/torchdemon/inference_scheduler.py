from collections import defaultdict
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Dict, List
from uuid import UUID

import numpy as np

from torchdemon.inference_client import InferenceClient
from torchdemon.inference_model import InferenceModel
from torchdemon.inference_queue import InferenceQueue
from torchdemon.models import (
    INFERENCE_DATA_T,
    InferencePayload,
    InferenceResult,
    Signal,
)


class InferenceScheduler:
    def __init__(
        self, inference_model: InferenceModel, inference_queue: InferenceQueue
    ):
        self._inference_model = inference_model
        self._inference_queue = inference_queue
        self._connections: Dict[UUID, Connection] = {}

    def create_client(self) -> InferenceClient:
        server_connection, client_connection = Pipe()
        inference_client = InferenceClient(client_connection)
        self._connections[inference_client.client_id] = server_connection
        return inference_client

    def check(self) -> None:
        closed_clients = []
        for _client_id, connection in self._connections.items():
            if connection.poll():
                inference_request = connection.recv()

                if inference_request.data == Signal.CLOSE:
                    self._connections[inference_request.client_id].close()
                    closed_clients.append(inference_request.client_id)
                    continue

                if inference_payloads_batch := self._inference_queue.process_request(
                    inference_payload=InferencePayload(
                        client_id=inference_request.client_id,
                        data=inference_request.data,
                    )
                ):
                    inference_results = self._run_inference(inference_payloads_batch)
                    for inference_result in inference_results:
                        self._connections[inference_result.client_id].send(
                            inference_result.data
                        )

        if inference_payloads_batch := self._inference_queue.check_wait():
            inference_results = self._run_inference(inference_payloads_batch)
            for inference_result in inference_results:
                self._connections[inference_result.client_id].send(
                    inference_result.data
                )

        self._connections = {
            client_id: connection
            for client_id, connection in self._connections.items()
            if client_id not in closed_clients
        }

    def connections_open(self) -> bool:
        return bool(self._connections)

    def _run_inference(
        self, inference_payloads_batch: List[InferencePayload]
    ) -> List[InferenceResult]:
        inference_data = defaultdict(list)
        for inference_payload in inference_payloads_batch:
            for k, ndarr in inference_payload.data.items():
                inference_data[k].append(ndarr)

        batched_inputs = {k: np.vstack(ndarrs) for k, ndarrs in inference_data.items()}
        batched_outputs = self._inference_model.infer(**batched_inputs)

        inference_results = []
        for i, inference_payload in enumerate(inference_payloads_batch):
            response_data: INFERENCE_DATA_T = {}
            for k, ndarr in batched_outputs.items():
                response_data[k] = ndarr[i]
            inference_result = InferenceResult(
                client_id=inference_payload.client_id, data=response_data
            )
            inference_results.append(inference_result)
        return inference_results
