"""A client manager that guarantees deterministic client sampling."""

import logging
import random
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class DeterministicClientManager(SimpleClientManager):
    """A deterministic client manager.

    Samples clients in the same order every time based on the seed. Also allows sampling
    with replacement.
    """

    def __init__(
        self,
        seed: int,
        enable_resampling: bool = False,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.rng = random.Random(seed)
        self.enable_resampling = enable_resampling

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        cids = list(self.clients)

        if criterion is not None:
            cids = [cid for cid in cids if criterion.select(self.clients[cid])]
        # Shuffle the list of clients

        available_cids = []
        if num_clients <= len(cids):
            available_cids = self.rng.sample(cids, num_clients)
        elif self.enable_resampling:
            available_cids = self.rng.choices(cids, k=num_clients)
        else:
            log(
                logging.INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(cids),
                num_clients,
            )
            available_cids = []

        client_list = [self.clients[cid] for cid in available_cids]
        log(logging.INFO, "Sampled the following clients: %s", available_cids)
        return client_list
