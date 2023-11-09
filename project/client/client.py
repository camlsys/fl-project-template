"""The default client implementation."""
from typing import Callable

import flwr as fl


class Client(fl.client.NumPyClient):
    """Virtual client for ray."""

    def __init__(self, cid: int) -> None:
        super().__init__()
        self.cid = cid


def get_client_generator() -> Callable[[int], Client]:
    """Return a function which creates a new Client for a given config."""
    return lambda i: Client(i)
