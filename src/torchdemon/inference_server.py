from multiprocessing.connection import Connection, Pipe
from typing import Dict, Optional
from uuid import UUID

import torch

from torchdemon.inference_client import InferenceClient
from torchdemon.inference_model import InferenceModel
from torchdemon.inference_queue import InferenceQueue
from torchdemon.inference_scheduler import InferenceScheduler
from torchdemon.models import TORCH_MODEL_T, InferencePayload, Signal


class InferenceServer:
    def __init__(
        self,
        model: TORCH_MODEL_T,
        batch_size: Optional[int] = None,
        max_wait_ns: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self._inference_scheduler = InferenceScheduler(
            inference_model=InferenceModel(model, device),
            inference_queue=InferenceQueue(batch_size, max_wait_ns),
        )
        self._connections: Dict[UUID, Connection] = {}

    def create_client(self) -> InferenceClient:
        server_connection, client_connection = Pipe()
        inference_client = InferenceClient(client_connection)
        self._connections[inference_client.client_id] = server_connection
        return inference_client

    def run(self) -> None:
        while self._connections:
            self.check()

    def check(self) -> None:
        closed_clients = set()
        for _, connection in self._connections.items():
            if connection.poll():
                inference_request = connection.recv()
                if inference_request.data == Signal.CLOSE:
                    self._connections[inference_request.client_id].close()
                    closed_clients.add(inference_request.client_id)
                    continue

                if inference_results := self._inference_scheduler.check(
                    inference_payload=InferencePayload(
                        client_id=inference_request.client_id,
                        data=inference_request.data,
                    )
                ):
                    for inference_result in inference_results:
                        self._connections[inference_result.client_id].send(
                            inference_result
                        )

        self._connections = {
            client_id: connection
            for client_id, connection in self._connections.items()
            if client_id not in closed_clients
        }

    def connections_open(self) -> bool:
        return bool(self._connections)
