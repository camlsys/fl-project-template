"""Default training and testing functions, local and federated."""

from collections.abc import Sized
from pathlib import Path
from typing import cast

from omegaconf import DictConfig
import torch
from flwr.common import NDArrays
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader

from project.client.client import ClientConfig
from project.fed.utils.utils import generic_set_parameters
from project.types.common import (
    FedDataloaderGen,
    FedEvalFN,
    IsolatedRNG,
    NetGen,
    OnFitConfigFN,
    TestFunc,
)
from project.utils.utils import obtain_device


class TrainConfig(BaseModel):
    """Training configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device
    # epochs: int
    # learning_rate: float

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def train(
    net: nn.Module | NDArrays,
    trainloader: DataLoader | None,
    _config: dict,
    _working_dir: Path,
    rng_tuple: IsolatedRNG,
    _hydra_config: DictConfig | None,
) -> tuple[int, dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    _config : Dict
        The configuration for the training.
        Contains the device, number of epochs and learning rate.
        Static type checking is done by the TrainConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if not isinstance(net, nn.Module) or trainloader is None:
        raise ValueError(
            "The default config does not use an implicit network generator/dataset"
        )

    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError(
            "Trainloader can't be 0, exiting...",
        )

    config: TrainConfig = TrainConfig(**_config)
    del _config

    net.to(config.device)
    net.train()

    return len(cast(Sized, trainloader.dataset)), {}


class TestConfig(BaseModel):
    """Testing configuration, allows '.' member access and static checking.

    Guarantees that all necessary components are present, fails early if config is
    mismatched to client.
    """

    device: torch.device

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


def test(
    net: nn.Module | NDArrays,
    testloader: DataLoader | None,
    _config: dict,
    _working_dir: Path,
    rng_tuple: IsolatedRNG,
    _hydra_config: DictConfig | None,
) -> tuple[float, int, dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    _config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.
    _working_dir : Path
        The working directory for the training.
        Unused.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    if not isinstance(net, nn.Module) or testloader is None:
        raise ValueError(
            "The default config does not use an implicit network generator/dataset"
        )

    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError(
            "Testloader can't be 0, exiting...",
        )

    config: TestConfig = TestConfig(**_config)
    del _config

    net.to(config.device)
    net.eval()

    return (
        0.0,
        len(cast(Sized, testloader.dataset)),
        {},
    )


def get_fed_eval_fn(
    net_generator: NetGen | None,
    fed_dataloader_generator: FedDataloaderGen | None,
    test_func: TestFunc,
    _config: dict,
    working_dir: Path,
    rng_tuple: IsolatedRNG,
    hydra_config: DictConfig | None,
) -> FedEvalFN | None:
    """Get the federated evaluation function.

    Parameters
    ----------
    net_generator : NetGenerator
        The function to generate the network.
    fed_dataloader_generator : DataLoader
        The DataLoader containing the data to test the network on.
    test_func : TestFunc
        The function to evaluate the network.
    _config : Dict
        The configuration for the testing.
        Contains the device.
        Static type checking is done by the TestConfig class.
    working_dir : Path
        The working directory for the training.
    _rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Optional[FedEvalFN]
        The evaluation function for the server
        if the testloader is not empty, else None.
    """
    config: ClientConfig = ClientConfig(**_config)
    del _config

    testloader = (
        fed_dataloader_generator(
            True,
            config.dataloader_config,
            rng_tuple,
            hydra_config,
        )
        if fed_dataloader_generator
        else None
    )

    def fed_eval_fn(
        _server_round: int,
        parameters: NDArrays,
        fake_config: dict,
    ) -> tuple[float, dict] | None:
        """Evaluate the model on the given data.

        Parameters
        ----------
        server_round : int
            The current server round.
        parameters : NDArrays
            The parameters of the model to evaluate.
        _config : Dict
            The configuration for the evaluation.

        Returns
        -------
        Optional[Tuple[float, Dict]]
            The loss and the accuracy of the input model on the given data.
        """
        net = (
            net_generator(config.net_config, rng_tuple, hydra_config)
            if net_generator
            else None
        )
        if net is not None:
            generic_set_parameters(net, parameters)

        config.run_config["device"] = obtain_device()

        if testloader is not None and len(cast(Sized, testloader.dataset)) == 0:
            return None

        loss, _num_samples, metrics = test_func(
            net if net is not None else parameters,
            testloader,
            config.run_config,
            working_dir,
            rng_tuple,
            hydra_config,
        )
        return loss, metrics

    return fed_eval_fn


def get_on_fit_config_fn(fit_config: dict) -> OnFitConfigFN:
    """Generate on_fit_config_fn based on a dict from the hydra config,.

    Parameters
    ----------
    fit_config : Dict
        The configuration for the fit function.
        Loaded dynamically from the config file.
    rng_tuple : IsolatedRNGTuple
        The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    Optional[OnFitConfigFN]
        The on_fit_config_fn for the server if the fit_config is not empty, else None.
    """
    # Fail early if the fit_config does not match expectations
    ClientConfig(**fit_config)

    def fit_config_fn(server_round: int) -> dict:
        """MNIST on_fit_config_fn.

        Parameters
        ----------
        server_round : int
            The current server round.
            Passed to the client

        Returns
        -------
        Dict
            The configuration for the fit function.
            Loaded dynamically from the config file.
        """
        fit_config["extra"]["server_round"] = server_round
        return fit_config

    return fit_config_fn


# Differences between the two will come
# from the config file
get_on_evaluate_config_fn = get_on_fit_config_fn
