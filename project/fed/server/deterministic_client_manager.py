"""A client manager that guarantees deterministic client sampling."""

import logging
import random
from typing import Any

from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from omegaconf import DictConfig


class DeterministicClientManager(SimpleClientManager):
    """A deterministic client manager.

    Samples clients in the same order every time based on the seed. Also allows sampling
    with replacement.
    """

    def __init__(
        self,
        client_cid_generator: random.Random,
        hydra_config: DictConfig | None,
        enable_resampling: bool = False,
    ) -> None:
        """Initialize DeterministicClientManager.

        Parameters
        ----------
        client_cid_generator : random.Random
            A random number generator to generate client cids.
        enable_resampling : bool
            Whether to allow sampling with replacement.

        Returns
        -------
        None
        """
        super().__init__()

        self.client_cid_generator = client_cid_generator
        self.enable_resampling = enable_resampling
        self.hydra_config = hydra_config

    def sample(
        self,
        num_clients: int,
        min_num_clients: int | None = None,
        criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        """Sample a number of Flower ClientProxy instances.

        Guarantees deterministic client sampling and enables
        sampling with replacement.

        Parameters
        ----------
        num_clients : int
            The number of clients to sample.
        min_num_clients : Optional[int]
            The minimum number of clients to sample.
        criterion : Optional[Criterion]
            A criterion to select clients.

        Returns
        -------
        List[ClientProxy]
            A list of sampled clients.
        """
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
            available_cids = self.client_cid_generator.sample(
                cids,
                num_clients,
            )
        elif self.enable_resampling:
            available_cids = self.client_cid_generator.choices(
                cids,
                k=num_clients,
            )
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
        log(
            logging.INFO,
            "Sampled the following clients: %s",
            available_cids,
        )

        return client_list


def dispatch_deterministic_client_manager(
    cfg: DictConfig, **kwargs: Any
) -> type[DeterministicClientManager] | None:
    """Dispatch the get_client_manager function based on the hydra config.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the get_client_manager function.
        Loaded dynamically from the config file.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the get_client_manager function.

    Returns
    -------
    type[DeterministicClientManager]
        The get_client_manager function.
    """
    client_manager: str | None = cfg.get("task", None).get("client_manager", None)

    if client_manager is None:
        return None

    if client_manager.upper() == "DEFAULT":
        return DeterministicClientManager

    return None
