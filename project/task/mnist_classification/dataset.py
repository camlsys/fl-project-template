"""MNIST dataset utilities for federated learning."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from project.task.default.dataset import (
    ClientDataloaderConfig as DefaultClientDataloaderConfig,
)
from project.task.default.dataset import (
    FedDataloaderConfig as DefaultFedDataloaderConfig,
)
from project.types.common import ClientDataloaderGen, FedDataloaderGen

# Use defaults for this very simple dataset
# Requires only batch size
ClientDataloaderConfig = DefaultClientDataloaderConfig
FedDataloaderConfig = DefaultFedDataloaderConfig


def get_dataloader_generators(
    partition_dir: Path,
) -> tuple[ClientDataloaderGen, FedDataloaderGen]:
    """Return a function that loads a client's dataset.

    Parameters
    ----------
    partition_dir : Path
        The path to the partition directory.
        Containing the training data of clients.
        Partitioned by client id.

    Returns
    -------
    Tuple[ClientDataloaderGen, FedDataloaderGen]
        A tuple of functions that return a DataLoader for a client's dataset
        and a DataLoader for the federated dataset.
    """

    def get_client_dataloader(cid: str | int, test: bool, _config: dict) -> DataLoader:
        """Return a DataLoader for a client's dataset.

        Parameters
        ----------
        cid : str|int
            The client's ID
        test : bool
            Whether to load the test set or not
        config : Dict
            The configuration for the dataset

        Returns
        -------
        DataLoader
            The DataLoader for the client's dataset
        """
        config: ClientDataloaderConfig = ClientDataloaderConfig(**_config)
        del _config

        client_dir = partition_dir / f"client_{cid}"
        if not test:
            dataset = torch.load(client_dir / "train.pt")
        else:
            dataset = torch.load(client_dir / "test.pt")
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=not test)

    def get_federated_dataloader(test: bool, _config: dict) -> DataLoader:
        """Return a DataLoader for federated train/test sets.

        Parameters
        ----------
        test : bool
            Whether to load the test set or not
        config : Dict
            The configuration for the dataset

        Returns
        -------
            DataLoader
            The DataLoader for the federated dataset
        """
        config: FedDataloaderConfig = FedDataloaderConfig(**_config)
        del _config

        if not test:
            return DataLoader(
                torch.load(partition_dir / "train.pt"),
                batch_size=config.batch_size,
                shuffle=not test,
            )
        else:
            return DataLoader(
                torch.load(partition_dir / "test.pt"),
                batch_size=config.batch_size,
                shuffle=not test,
            )

    return get_client_dataloader, get_federated_dataloader
