"""Dispatch the MNIST functionality to project.main.

The dispatch functions are used to
dynamically select the correct functions from the task
based on the hydra config file.
The following categories of functionality are grouped together:
    - train/test and fed test functions
    - net generator and dataloader generator functions
    - fit/eval config functions

The top-level project.dipatch module operates as a pipeline
and selects the first function which does not return None.

Do not throw any errors based on not finding a given attribute
in the configs under any circumstances.

If you cannot match the config file,
return None and the dispatch of the next task
in the chain specified by project.dispatch will be used.
"""

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from project.task.default.dispatch import dispatch_config as dispatch_default_config
from project.task.mnist_classification.dataset import get_dataloader_generators
from project.task.mnist_classification.models import get_logistic_regression, get_net
from project.task.mnist_classification.train_test import get_fed_eval_fn, test, train
from project.types.common import DataStructure, TrainStructure


def dispatch_train(cfg: DictConfig) -> Optional[TrainStructure]:
    """Dispatch the train/test and fed test functions based on the config file.

    Do not throw any errors based on not finding a given attribute
    in the configs under any circumstances.

    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the train function.
        Loaded dynamically from the config file.

    Returns
    -------
    Optional[TrainStructure]
        The train function, test function and the get_fed_eval_fn function.
        Return None if you cannot match the cfg.
    """
    # Select the value for the key with None default
    train_structure: Optional[str] = cfg.get("task", {}).get("train_structure", None)

    # Only consider not None and uppercase matches
    if train_structure is not None and train_structure.upper() == "MNIST":
        return train, test, get_fed_eval_fn

    # Cannot match, send to next dispatch in chain
    return None


def dispatch_data(cfg: DictConfig) -> Optional[DataStructure]:
    """Dispatch the train/test and fed test functions based on the config file.

    Do not throw any errors based on not finding a given attribute
    in the configs under any circumstances.

    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the data functions.
        Loaded dynamically from the config file.

    Returns
    -------
    Optional[DataStructure]
        The net generator, client dataloader generator and fed dataloader generator.
        Return None if you cannot match the cfg.
    """
    # Select the value for the key with {} default at nested dicts
    # and None default at the final key
    client_model_and_data: Optional[str] = cfg.get("task", {}).get(
        "model_and_data", None
    )

    # Select the partition dir
    # if it does not exist data cannot be loaded
    # for MNIST and the dispatch should return None
    partition_dir: Optional[str] = cfg.get("dataset", {}).get("partition_dir", None)

    # Only consider situations where both are not None
    # otherwise data loading would failr later
    if client_model_and_data is not None and partition_dir is not None:
        # Obtain the dataloader generators
        # for the provided partition dir
        client_dataloader_gen, fed_dataloater_gen = get_dataloader_generators(
            Path(partition_dir)
        )

        # Case insensitive matches
        if client_model_and_data.upper() == "MNIST_CNN":
            return get_net, client_dataloader_gen, fed_dataloater_gen
        elif client_model_and_data.upper() == "MNIST_LR":
            return get_logistic_regression, client_dataloader_gen, fed_dataloater_gen

    # Cannot match, send to next dispatch in chain
    return None


dispatch_config = dispatch_default_config
