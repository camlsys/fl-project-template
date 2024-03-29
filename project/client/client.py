"""The default client implementation.

Make sure the model and dataset are not loaded before the fit function.
"""

import random
from pathlib import Path

import flwr as fl
from flwr.common import NDArrays
from omegaconf import DictConfig
from pydantic import BaseModel
from torch import nn

from project.fed.utils.utils import (
    generic_get_parameters,
    generic_set_parameters,
    get_isolated_rng_tuple,
)
from project.types.common import (
    CID,
    ClientDataloaderGen,
    ClientGen,
    EvalRes,
    FitRes,
    NetGen,
    TestFunc,
    TrainFunc,
)
from project.utils.utils import obtain_device


class ClientConfig(BaseModel):
    """Fit/eval config, allows '.' member access and static checking.

    Used to check whether each component has its own independent config present. Each
    component should then use its own Pydantic model to validate its config. For
    anything extra, use the extra field as a simple dict.
    """

    # Instantiate model
    net_config: dict
    # Instantiate dataloader
    dataloader_config: dict
    # For train/test
    run_config: dict
    # Additional params used like a Dict
    extra: dict

    class Config:
        """Setting to allow any types, including library ones like torch.device."""

        arbitrary_types_allowed = True


class Client(fl.client.NumPyClient):
    """Virtual client for ray."""

    def __init__(
        self,
        cid: CID,
        working_dir: Path,
        net_generator: NetGen | None,
        dataloader_gen: ClientDataloaderGen | None,
        train: TrainFunc,
        test: TestFunc,
        client_seed: int,
        hydra_config: DictConfig | None,
    ) -> None:
        """Initialize the client.

        Only ever instantiate the model or load dataset
        inside fit/eval, never in init.

        Parameters
        ----------
        cid : int | str | Path
            The client's ID.
        working_dir : Path
            The path to the working directory.
        net_generator : NetGen
            The network generator.
        dataloader_gen : ClientDataloaderGen
            The dataloader generator.
            Uses the client id to determine partition.

        Returns
        -------
        None
        """
        super().__init__()
        self.cid = cid
        self.net_generator = net_generator
        self.working_dir = working_dir
        self.net: nn.Module | None = None
        self.dataloader_gen = dataloader_gen
        self.train = train
        self.test = test

        # For deterministic client execution
        # The client_seed is generated from a specific Generator
        self.client_seed = client_seed
        self.rng_tuple = get_isolated_rng_tuple(self.client_seed, obtain_device())

        self.hydra_config = hydra_config

    def fit(
        self,
        parameters: NDArrays,
        _config: dict,
    ) -> FitRes:
        """Fit the model using the provided parameters.

        Only ever instantiate the model or load dataset
        inside fit, never in init.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to use for training.
        _config : Dict
            The configuration for the training.
            Uses the pydantic model for static checking.

        Returns
        -------
        FitRes
            The parameters after training, the number of samples used and the metrics.
        """
        config: ClientConfig = ClientConfig(**_config)
        del _config

        config.run_config["device"] = obtain_device()

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )
        trainloader = (
            self.dataloader_gen(
                self.cid,
                False,
                config.dataloader_config,
                self.rng_tuple,
                self.hydra_config,
            )
            if self.dataloader_gen is not None
            else None
        )
        num_samples, metrics = self.train(
            self.net,
            trainloader,
            config.run_config,
            self.working_dir,
            self.rng_tuple,
            self.hydra_config,
        )

        return (
            self.get_parameters({}),
            num_samples,
            metrics,
        )

    def evaluate(
        self,
        parameters: NDArrays,
        _config: dict,
    ) -> EvalRes:
        """Evaluate the model using the provided parameters.

        Only ever instantiate the model or load dataset
        inside eval, never in init.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to use for evaluation.
        _config : Dict
            The configuration for the evaluation.
            Uses the pydantic model for static checking.

        Returns
        -------
        EvalRes
            The loss, the number of samples used and the metrics.
        """
        config: ClientConfig = ClientConfig(**_config)
        del _config

        config.run_config["device"] = obtain_device()

        self.net = self.set_parameters(
            parameters,
            config.net_config,
        )
        testloader = (
            self.dataloader_gen(
                self.cid,
                True,
                config.dataloader_config,
                self.rng_tuple,
                self.hydra_config,
            )
            if self.dataloader_gen is not None
            else None
        )
        loss, num_samples, metrics = self.test(
            self.net,
            testloader,
            config.run_config,
            self.working_dir,
            self.rng_tuple,
            self.hydra_config,
        )
        return loss, num_samples, metrics

    def get_parameters(self, config: dict) -> NDArrays:
        """Obtain client parameters.

        If the network is currently none,generate a network using the net_generator.

        Parameters
        ----------
        config : Dict
            The configuration for the training.

        Returns
        -------
        NDArrays
            The parameters of the network.
        """
        if self.net is None:
            except_str: str = """Network is None.
                Call set_parameters first and
                do not use this template without a get_initial_parameters function.
            """
            raise ValueError(
                except_str,
            )

        return generic_get_parameters(self.net)

    def set_parameters(
        self,
        parameters: NDArrays,
        config: dict,
    ) -> nn.Module:
        """Set client parameters.

        First generated the network. Only call this in fit/eval.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to set.
        config : Dict
            The configuration for the network generator.

        Returns
        -------
        nn.Module
            The network with the new parameters.
        """
        net = (
            self.net_generator(config, self.rng_tuple, self.hydra_config)
            if self.net_generator is not None
            else None
        )
        if net is None:
            raise ValueError("Cannot request parameters without net generator method")

        generic_set_parameters(
            net,
            parameters,
            to_copy=False,
        )
        return net

    def __repr__(self) -> str:
        """Implement the string representation based on cid."""
        return f"Client(cid={self.cid})"

    def get_properties(self, config: dict) -> dict:
        """Implement how to get properties."""
        return {}


def get_client_generator(
    working_dir: Path,
    net_generator: NetGen | None,
    dataloader_gen: ClientDataloaderGen | None,
    train: TrainFunc,
    test: TestFunc,
    client_seed_generator: random.Random,
    hydra_config: DictConfig | None,
) -> ClientGen:
    """Return a function which creates a new Client.

    Client has access to the working dir,
    can generate a network and can generate a dataloader.
    The client receives train and test functions with pre-defined APIs.

    Parameters
    ----------
    working_dir : Path
        The path to the working directory.
    net_generator : NetGen
        The network generator.
        Please respect the pydantic schema.
    dataloader_gen : ClientDataloaderGen
        The dataloader generator.
        Uses the client id to determine partition.
        Please respect the pydantic schema.
    train : TrainFunc
        The train function.
        Please respect the interface and pydantic schema.
    test : TestFunc
        The test function.
        Please respect the interface and pydantic schema.
    seed : int
        The global seed for the random number generators.
    random_state : tuple[Any,Any,Any]
        The random state for the random number generator.
    np_random_state : dict[str,Any]
        The numpy random state for the random number generator.
    torch_random_state : torch.Tensor

    Returns
    -------
    ClientGen
        The function which creates a new Client.
    """

    def client_generator(cid: CID) -> Client:
        """Return a new Client.

        Parameters
        ----------
        cid : int | str | Path
            The client's ID.

        Returns
        -------
        Client
            The new Client.
        """
        return Client(
            cid,
            working_dir,
            net_generator,
            dataloader_gen,
            train,
            test,
            client_seed=client_seed_generator.randint(0, 2**32 - 1),
            hydra_config=hydra_config,
        )

    return client_generator
