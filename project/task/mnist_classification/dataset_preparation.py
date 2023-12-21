"""Functions for MNIST download and processing."""

import logging
from collections.abc import Sequence, Sized
from pathlib import Path
from typing import cast

import hydra
import numpy as np
import torch
from flwr.common.logger import log
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


def _download_data(
    dataset_dir: Path,
) -> tuple[MNIST, MNIST]:
    """Download (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ],
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    trainset = MNIST(
        str(dataset_dir),
        train=True,
        download=True,
        transform=transform,
    )
    testset = MNIST(
        str(dataset_dir),
        train=False,
        download=True,
        transform=transform,
    )
    return trainset, testset


# pylint: disable=too-many-locals
def _partition_data(  # pylint: disable=too-many-arguments
    trainset: MNIST,
    testset: MNIST,
    num_clients: int,
    seed: int,
    iid: bool,
    power_law: bool,
    balance: bool,
) -> tuple[list[Subset] | list[ConcatDataset], MNIST]:
    """Split training set into iid or non iid partitions to simulate the federated.

    setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed
        by chunks to each client (used to test the convergence in a worst case scenario)
        , by default False
    power_law: bool
        Whether to follow a power-law distribution when assigning number of samples
        for each client, defaults to True
    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default False
    seed : int
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[MNIST], MNIST]
        A list of dataset for each client and a single dataset to be used for testing
        the model.
    """
    if balance:
        trainset = _balance_classes(trainset, seed)

    partition_size = int(
        len(cast(Sized, trainset)) / num_clients,
    )
    lengths = [partition_size] * num_clients

    if iid:
        datasets = random_split(
            trainset,
            lengths,
            torch.Generator().manual_seed(seed),
        )
    elif power_law:
        trainset_sorted = _sort_by_class(trainset)
        datasets = _power_law_split(
            trainset_sorted,
            num_partitions=num_clients,
            num_labels_per_partition=2,
            min_data_per_partition=10,
            mean=0.0,
            sigma=2.0,
        )
    else:
        shard_size = int(partition_size / 2)
        idxs = trainset.targets.argsort()
        sorted_data = Subset(
            trainset,
            cast(Sequence[int], idxs),
        )
        tmp = []
        for idx in range(num_clients * 2):
            tmp.append(
                Subset(
                    sorted_data,
                    cast(
                        Sequence[int],
                        np.arange(
                            shard_size * idx,
                            shard_size * (idx + 1),
                        ),
                    ),
                ),
            )
        idxs_list = torch.randperm(
            num_clients * 2,
            generator=torch.Generator().manual_seed(seed),
        )
        datasets = [
            ConcatDataset(
                (
                    tmp[idxs_list[2 * i]],
                    tmp[idxs_list[2 * i + 1]],
                ),
            )
            for i in range(num_clients)
        ]

    return datasets, testset


def _balance_classes(
    trainset: MNIST,
    seed: int,
) -> MNIST:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : MNIST
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    MNIST
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [
        Subset(
            trainset,
            cast(Sequence[int], idxs[: int(smallest)]),
        ),
    ]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(
                trainset,
                cast(
                    Sequence[int],
                    idxs[int(count) : int(count + smallest)],
                ),
            ),
        )
        tmp_targets.append(
            trainset.targets[idxs[int(count) : int(count + smallest)]],
        )
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled),
        generator=torch.Generator().manual_seed(seed),
    )
    shuffled = cast(
        MNIST,
        Subset(
            unshuffled,
            cast(Sequence[int], shuffled_idxs),
        ),
    )
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled


def _sort_by_class(
    trainset: MNIST,
) -> MNIST:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : MNIST
        The training dataset that needs to be sorted.

    Returns
    -------
    MNIST
        The sorted training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    idxs = trainset.targets.argsort()  # sort targets in ascending order

    tmp = []  # create subset of smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(
                trainset,
                cast(
                    Sequence[int],
                    idxs[start : int(count + start)],
                ),
            ),
        )  # add rest of classes
        tmp_targets.append(
            trainset.targets[idxs[start : int(count + start)]],
        )
        start += count
    sorted_dataset = cast(
        MNIST,
        ConcatDataset(tmp),
    )  # concat dataset
    sorted_dataset.targets = torch.cat(
        tmp_targets,
    )  # concat targets
    return sorted_dataset


# pylint: disable=too-many-locals, too-many-arguments
def _power_law_split(  # pylint: disable=too-many-arguments
    sorted_trainset: MNIST,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> list[Subset]:
    """Partition the dataset following a power-law distribution. It follows the.

    implementation of Li et al 2020: https://arxiv.org/abs/1812.06127 with default
    values set accordingly.

    Parameters
    ----------
    sorted_trainset : MNIST
        The training dataset sorted by label/class.
    num_partitions: int
        Number of partitions to create
    num_labels_per_partition: int
        Number of labels to have in each dataset partition. For
        example if set to two, this means all training examples in
        a given partition will be long to the same two classes. default 2
    min_data_per_partition: int
        Minimum number of datapoints included in each partition, default 10
    mean: float
        Mean value for LogNormal distribution to construct power-law, default 0.0
    sigma: float
        Sigma value for LogNormal distribution to construct power-law, default 2.0

    Returns
    -------
    MNIST
        The partitioned training dataset.
    """
    targets = sorted_trainset.targets
    full_idx = list(range(len(targets)))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()

    partitions_idx: list[list[int]] = []
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(
        min_data_per_partition / num_labels_per_partition,
    )
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            # label for the u_id-th client
            cls = (u_id + cls_idx) % num_classes
            # record minimum data
            indices = list(
                full_idx[
                    labels_cs[cls]
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ],
            )
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # add remaining images following power-law
    probs = np.random.lognormal(
        mean,
        sigma,
        (
            num_classes,
            int(num_partitions / num_classes),
            num_labels_per_partition,
        ),
    )
    remaining_per_class = class_counts - hist
    # obtain how many samples each partition should be assigned for each of the
    # labels it contains
    # pylint: disable=too-many-function-args
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(
                probs[cls, u_id // num_classes, cls_idx],
            )

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct partition subsets
    return [Subset(sorted_trainset, p) for p in partitions_idx]


@hydra.main(
    config_path="../../conf",
    config_name="mnist",
    version_base=None,
)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Download and preprocess the dataset.

    Please include here all the logic
    Please use the Hydra config style as much as possible specially
    for parts that can be customised (e.g. how data is partitioned)

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    # Download the dataset
    trainset, testset = _download_data(
        Path(cfg.dataset.dataset_dir),
    )

    # Partition the dataset
    # ideally, the fed_test_set can be composed in three ways:
    # 1. fed_test_set = centralised test set like MNIST
    # 2. fed_test_set = concatenation of all test sets of all clients
    # 3. fed_test_set = test sets of reserved unseen clients
    client_datasets, fed_test_set = _partition_data(
        trainset,
        testset,
        cfg.dataset.num_clients,
        cfg.dataset.seed,
        cfg.dataset.iid,
        cfg.dataset.power_law,
        cfg.dataset.balance,
    )

    # 2. Save the datasets
    # unnecessary for this small dataset, but useful for large datasets
    partition_dir = Path(cfg.dataset.partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    # Save the centralised test set
    # a centrailsed training set would also be possible
    # but is not used here
    torch.save(fed_test_set, partition_dir / "test.pt")

    # Save the client datasets
    for idx, client_dataset in enumerate(client_datasets):
        client_dir = partition_dir / f"client_{idx}"
        client_dir.mkdir(parents=True, exist_ok=True)

        len_val = int(
            len(client_dataset) / (1 / cfg.dataset.val_ratio),
        )
        lengths = [len(client_dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            client_dataset,
            lengths,
            torch.Generator().manual_seed(cfg.dataset.seed),
        )
        # Alternative would have been to create train/test split
        # when the dataloader is instantiated
        torch.save(ds_train, client_dir / "train.pt")
        torch.save(ds_val, client_dir / "test.pt")


if __name__ == "__main__":
    download_and_preprocess()
