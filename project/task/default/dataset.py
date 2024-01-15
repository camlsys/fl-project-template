"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders
for your dataset (for both the clients and server).
If you are using a custom dataset class, this module is the place to define it.
If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module.
You can use all those functions from functions/methods defined here of course.
"""

from collections import defaultdict

import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset, TensorDataset

from project.types.common import CID, IsolatedRNG


class ClientDataloaderConfig(BaseModel):
    """Dataloader configuration for the client.

    Allows '.' member access and static checking. Guarantees that all necessary
    components are present, fails early if config is mismatched to dataloader.
    """

    batch_size: int

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


class FedDataloaderConfig(BaseModel):
    """Dataloader configuration for the client.

    Allows '.' member access and static checking. Guarantees that all necessary
    components are present, fails early if config is mismatched to dataloader.
    """

    batch_size: int

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def get_client_dataloader(
    cid: CID,
    test: bool,
    _config: dict,
    _rng_tuple: IsolatedRNG,
) -> DataLoader:
    """Return a DataLoader for a client's dataset.

    Parameters
    ----------
    cid : str|int
        The client's ID
    test : bool
        Whether to load the test set or not
    cfg : Dict
        The configuration for the dataset
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    DataLoader
        The DataLoader for the client's dataset
    """
    # Create an empty TensorDataset for illustration purposes
    config: ClientDataloaderConfig = ClientDataloaderConfig(
        **_config,
    )
    del _config

    # You should load/create one train/test dataset per client
    if not test:
        empty_trainset_dict: dict[
            CID,
            Dataset,
        ] = defaultdict(
            lambda: TensorDataset(
                torch.Tensor([1]),
                torch.Tensor([1]),
            ),
        )
        # Choose the client dataset based on the client id and train/test
        dataset = empty_trainset_dict[cid]
    else:
        empty_testset_dict: dict[
            CID,
            Dataset,
        ] = defaultdict(
            lambda: TensorDataset(
                torch.Tensor([1]),
                torch.Tensor([1]),
            ),
        )
        # Choose the client dataset based on the client id and train/test
        dataset = empty_testset_dict[cid]

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=not test,
        drop_last=True,
    )


def get_fed_dataloader(
    test: bool,
    _config: dict,
    _rng_tuple: IsolatedRNG,
) -> DataLoader:
    """Return a DataLoader for federated train/test sets.

    Parameters
    ----------
    test : bool
        Whether to load the test set or not
    config : Dict
        The configuration for the dataset
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
        DataLoader
        The DataLoader for the federated dataset
    """
    config: FedDataloaderConfig = FedDataloaderConfig(
        **_config,
    )
    del _config

    # Create one train/test empty dataset for the server
    if not test:
        empty_trainset: Dataset = TensorDataset(
            torch.Tensor([1]),
            torch.Tensor([1]),
        )
        # Choose the server dataset based on the train/test
        dataset = empty_trainset
    else:
        empty_testset: Dataset = TensorDataset(
            torch.Tensor([1]),
            torch.Tensor([1]),
        )
        # Choose the server dataset based on the train/test
        dataset = empty_testset

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=not test,
        drop_last=True,
    )
