"""MNIST dataset utilities for federated learning."""


from pathlib import Path
from typing import Callable, Dict

import torch
from torch.utils.data import DataLoader


def get_load_client_dataloader(
    partition_dir: Path,
) -> Callable[[str | int, bool, Dict], DataLoader]:
    """Return a function that loads a client's dataset."""

    def get_dataloader(cid: str | int, test: bool, cfg: Dict) -> DataLoader:
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

        return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=not test)

    return get_dataloader
