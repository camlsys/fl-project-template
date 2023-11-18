"""The default client implementation.

Make sure the model and dataset are not loaded before the fit function.
"""
from pathlib import Path
from typing import Dict, Optional

import flwr as fl
import torch.nn as nn
from flwr.common import NDArrays

from project.fed.utils.utils import generic_get_parameters, generic_set_parameters
from project.types.common import ClientDataloaderGen, ClientGen, NetGen


class Client(fl.client.NumPyClient):
    """Virtual client for ray."""

    def __init__(
        self,
        cid: int | str,
        working_dir: Path,
        net_generator: NetGen,
        dataloader_gen: ClientDataloaderGen,
    ) -> None:
        """Initialize the client.

        Only ever instantiate the model or load dataset
        inside fit/eval, never in init.

        Parameters
        ----------
        cid : int | str
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
        self.net: Optional[nn.Module] = None
        self.dataloader_gen = dataloader_gen

    # def fit(
    #     self,
    #     parameters: NDArrays,
    #     config: Dict,
    # ) -> FitRes:
    #     """Fit the model using the provided parameters.

    #     Only ever instantiate the model or load dataset
    #     inside fit, never in init.
    #     Parameters
    #     ----------
    #     parameters : NDArrays
    #         The parameters to use for training.
    #     config : Dict
    #         The configuration for the training.

    #     Returns
    #     -------
    #     FitRes
    #         The parameters after training, the number of samples used and the metrics.
    #     """
    #     self.net = self.set_parameters(parameters,  config)

    # def evaluate(
    #     self,
    #     parameters: NDArrays,
    #     config: Dict,
    # ) -> EvalRes:
    #     """Evaluate the model using the provided parameters.

    #     Only ever instantiate the model or load dataset
    #     inside eval, never in init.
    #     Parameters
    #     ----------
    #     parameters : NDArrays
    #         The parameters to use for evaluation.
    #     config : Dict
    #         The configuration for the evaluation.

    #     Returns
    #     -------
    #     EvalRes
    #         The loss, the number of samples used and the metrics.
    #     """
    #     self.net = self.set_parameters(parameters,  config)

    def get_parameters(self, config: Dict) -> NDArrays:
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
            return generic_get_parameters(self.net_generator(config))
        return generic_get_parameters(self.net)

    def set_parameters(self, parameters: NDArrays, config: Dict) -> nn.Module:
        """Set client parameters.

        First generated the network. Only call this in fit/eval.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to set.
        config : Dict
            The configuration for the training.

        Returns
        -------
        nn.Module
            The network with the new parameters.
        """
        net = self.net_generator(config)
        generic_set_parameters(net, parameters, to_copy=False)
        return net

    def __repr__(self) -> str:
        """Implement the string representation based on cid."""
        return f"Client(cid={self.cid})"

    def get_properties(self, config: Dict) -> Dict:
        """Implement how to get properties."""
        return {}


def get_client_generator(
    working_dir: Path, net_generator: NetGen, dataloader_gen: ClientDataloaderGen
) -> ClientGen:
    """Return a function which creates a new Client for a given config.

    Client has access to the working dir, can generate a network and
    can generate a dataloader.

    Parameters
    ----------
    working_dir : Path
        The path to the working directory.
    net_generator : NetGen
        The network generator.
    dataloader_gen : ClientDataloaderGen
        The dataloader generator.
        Uses the client id to determine partition.

    Returns
    -------
    Callable[[int | str], Client]
        The function which creates a new Client.
    """
    return lambda i: Client(
        i,
        working_dir=working_dir,
        net_generator=net_generator,
        dataloader_gen=dataloader_gen,
    )
