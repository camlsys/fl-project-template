"""MNIST dataset utilities for federated learning."""


from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from project.types.common import ClientDataloaderGen, FedDataloaderGen


def get_dataloader_generators(
    partition_dir: Path,
) -> Tuple[ClientDataloaderGen, FedDataloaderGen]:
    """Return a function that loads a client's dataset."""

    def get_client_dataloader(cid: str | int, test: bool, config: Dict) -> DataLoader:
        """Return a DataLoader for a client's dataset.

        Parameters
        ----------
        cid : str|int
            The client's ID
        test : bool
            Whether to load the test set or not
        cfg : Dict
            The configuration for the dataset

        Returns
        -------
        DataLoader
            The DataLoader for the client's dataset
        """
        client_dir = partition_dir / f"client_{cid}"
        if not test:
            dataset = torch.load(client_dir / "train.pt")
        else:
            dataset = torch.load(client_dir / "test.pt")
        return DataLoader(dataset, batch_size=config["batch_size"], shuffle=not test)

    def get_federated_dataloader(test: bool, config: Dict) -> DataLoader:
        """Return a DataLoader for federated train/test sets.

        Parameters
        ----------
        test : bool
            Whether to load the test set or not
            cfg : Dict
            The configuration for the dataset

        Returns
        -------
            DataLoader
            The DataLoader for the federated dataset
        """
        if not test:
            return DataLoader(
                torch.load(partition_dir / "train.pt"),
                batch_size=config["batch_size"],
                shuffle=not test,
            )
        else:
            return DataLoader(
                torch.load(partition_dir / "test.pt"),
                batch_size=config["batch_size"],
                shuffle=not test,
            )

    return get_client_dataloader, get_federated_dataloader
