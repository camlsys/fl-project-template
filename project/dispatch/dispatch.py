"""Dispatches the functionality of the task.

This gives us the ability to dynamically choose functionality based on the hydra dict
config without losing static type checking.
"""

from collections.abc import Callable
from typing import Any

from omegaconf import DictConfig

from project.task.default.dispatch import dispatch_config as dispatch_default_config
from project.task.default.dispatch import dispatch_data as dispatch_default_data
from project.task.default.dispatch import dispatch_train as dispatch_default_train
from project.client.client import (
    dispatch_client_gen as dispatch_default_client_gen,
)

from project.task.mnist_classification.dispatch import (
    dispatch_config as dispatch_mnist_config,
)
from project.task.mnist_classification.dispatch import (
    dispatch_data as dispatch_mnist_data,
)
from project.task.mnist_classification.dispatch import (
    dispatch_train as dispatch_mnist_train,
)
from project.types.common import (
    ConfigStructure,
    DataStructure,
    GetClientGen,
    TrainStructure,
)


def dispatch_train(cfg: DictConfig, **kwargs: Any) -> TrainStructure:
    """Dispatch the train/test and fed test functions based on the config file.

    Functionality should be added to the dispatch.py file in the task folder.
    Statically specify the new dispatch function in the list,
    function order determines precedence if two different tasks may match the config.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the train function.
        Loaded dynamically from the config file.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the train function.

    Returns
    -------
    TrainStructure
        The train function, test function and the get_fed_eval_fn function.
    """
    # Create the list of task dispatches to try
    task_train_functions: list[Callable[..., TrainStructure | None]] = [
        dispatch_default_train,
        dispatch_mnist_train,
    ]

    # Match the first function which does not return None
    for task in task_train_functions:
        result = task(cfg, **kwargs)
        if result is not None:
            return result

    raise ValueError(
        f"Unable to match the train/test and fed_test functions: {cfg}",
    )


def dispatch_data(cfg: DictConfig, **kwargs: Any) -> DataStructure:
    """Dispatch the net generator and dataloader client/fed generator functions.

    Functionality should be added to the dispatch.py file in the task folder.
    Statically specify the new dispatch function in the list,
    function order determines precedence if two different tasks may match the config.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the data function.
        Loaded dynamically from the config file.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the data function.

    Returns
    -------
    DataStructure
        The net generator and dataloader generator functions.
    """
    # Create the list of task dispatches to try
    task_data_dependent_functions: list[Callable[..., DataStructure | None]] = [
        dispatch_mnist_data,
        dispatch_default_data,
    ]

    # Match the first function which does not return None
    for task in task_data_dependent_functions:
        result = task(cfg, **kwargs)
        if result is not None:
            return result

    raise ValueError(
        f"Unable to match the net generator and dataloader generator functions: {cfg}",
    )


def dispatch_config(cfg: DictConfig, **kwargs: Any) -> ConfigStructure:
    """Dispatch the fit/eval config functions based on on the hydra config.

    Functionality should be added to the dispatch.py
    file in the task folder.
    Statically specify the new dispatch function in the list,
    function order determines precedence
    if two different tasks may match the config.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the config function.
        Loaded dynamically from the config file.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the config function.

    Returns
    -------
    ConfigStructure
        The config functions.
    """
    # Create the list of task dispatches to try
    task_config_functions: list[Callable[..., ConfigStructure | None]] = [
        dispatch_mnist_config,
        dispatch_default_config,
    ]

    # Match the first function which does not return None
    for task in task_config_functions:
        result = task(cfg, **kwargs)
        if result is not None:
            return result

    raise ValueError(
        f"Unable to match the config generation functions: {cfg}",
    )


def dispatch_get_client_generator(cfg: DictConfig, **kwargs: Any) -> GetClientGen:
    """Dispatch the get_client_generator function based on the hydra config.

    Functionality should be added to the dispatch.py
    file in the task folder.
    Statically specify the new dispatch function in the list,
    function order determines precedence
    if two different tasks may match the config.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the get_client_generators function.
        Loaded dynamically from the config file.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the get_client_generators function.

    Returns
    -------
    GetClientGen
        The get_client_generators function.
    """
    # Create the list of task dispatches to try
    task_get_client_generators: list[Callable[..., Callable | None]] = [
        dispatch_default_client_gen
    ]

    # Match the first function which does not return None
    for task in task_get_client_generators:
        result = task(cfg, **kwargs)
        if result is not None:
            return result

    raise ValueError(
        f"Unable to match the get_client_generators function: {cfg}",
    )
