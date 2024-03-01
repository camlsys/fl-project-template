"""Dispatch the functionality of the task to project.main.

The dispatch functions are used to dynamically select
the correct functions from the task
based on the hydra config file.
You need to write dispatch functions for three categories:
    - train/test and fed test functions
    - net generator and dataloader generator functions
    - fit/eval config functions

The top-level project.dispatch module operates as a pipeline
and selects the first function which does not return None.
Do not throw any errors based on not finding
a given attribute in the configs under any circumstances.
If you cannot match the config file,
return None and the dispatch of the next task
in the chain specified by project.dispatch will be used.
"""

from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from project.task.default.dataset import (
    get_client_dataloader,
    get_fed_dataloader,
    init_working_dir,
)
from project.task.default.models import get_net
from project.task.default.train_test import (
    get_fed_eval_fn,
    get_on_evaluate_config_fn,
    get_on_fit_config_fn,
    test,
    train,
)
from project.types.common import ConfigStructure, DataStructure, TrainStructure


def dispatch_train(
    cfg: DictConfig,
    **kwargs: Any,
) -> TrainStructure | None:
    """Dispatch the train/test and fed test functions based on the config file.

    Do not throw any errors based on not finding
    a given attribute in the configs under any circumstances.
    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the train function.
        Loaded dynamically from the config file.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the train function.

    Returns
    -------
    Optional[TrainStructure]
        The train function, test function and the get_fed_eval_fn function.
        Return None if you cannot match the cfg.
    """
    # Select the value for the key with None default
    train_structure: str | None = cfg.get("task", {}).get(
        "train_structure",
        None,
    )

    # Only consider not None matches, case insensitive
    if train_structure is not None and train_structure.upper() == "DEFAULT":
        return train, test, get_fed_eval_fn

    # Cannot match, send to next dispatch in chain
    return None


def dispatch_data(cfg: DictConfig, **kwargs: Any) -> DataStructure | None:
    """Dispatch the net and dataloader client/fed generator functions.

    Do not throw any errors based on not finding
    a given attribute in the configs under any circumstances.
    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the data functions.
        Loaded dynamically from the config file.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the data functions.

    Returns
    -------
    Optional[DataStructure]
        The net generator, client dataloader generator and fed dataloader generator.
        Return None if you cannot match the cfg.
    """
    # Select the value for the key with {} default at nested dicts
    # and None default at the final key
    client_model_and_data: str | None = cfg.get(
        "task",
        {},
    ).get("model_and_data", None)

    # Only consider not None matches, case insensitive
    if client_model_and_data is not None and client_model_and_data.upper() == "DEFAULT":
        return (
            get_net,
            get_client_dataloader,
            get_fed_dataloader,
            init_working_dir,
        )

    # Cannot match, send to next dispatch in chain
    return None


def dispatch_config(
    cfg: DictConfig,
    **kwargs: Any,
) -> ConfigStructure | None:
    """Dispatches the config function based on the config_structure in the config file.

    By default it simply takes the fit_config and evaluate_config
    dicts from the hydra config.
    Only change if a more complex behavior
    (such as varying the config across rounds) is needed.

    Do not throw any errors based on not finding
    a given attribute in the configs under any circumstances.
    If you cannot match the config file,
    return None and the dispatch of the next task
    in the chain specified by project.dispatch will be used.

    Parameters
    ----------
    cfg : DictConfig
        The configuration for the config function.
        Loaded dynamically from the config file.
    kwargs : dict[str, Any]
        Additional keyword arguments to pass to the config function.

    Returns
    -------
    Optional[ConfigStructure]
        The fit_config and evaluate_config functions.
        Return None if you cannot match the cfg.
    """
    # Select the values for the key with {} default at nested dicts
    # and None default at the final key
    fit_config: dict | None = cfg.get("task", {}).get(
        "fit_config",
        None,
    )
    eval_config: dict | None = cfg.get("task", {}).get(
        "eval_config",
        None,
    )

    # Only consider existing config dicts as matches
    if fit_config is not None and eval_config is not None:
        return get_on_fit_config_fn(
            cast(dict, OmegaConf.to_container(fit_config)),
        ), get_on_evaluate_config_fn(
            cast(dict, OmegaConf.to_container(eval_config)),
        )

    return None
