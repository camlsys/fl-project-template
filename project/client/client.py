"""The default client implementation.

Make sure the model and dataset are not loaded before the fit function.
"""
from pathlib import Path
from typing import Callable, Dict, Optional

import flwr as fl
import torch.nn as nn
from flwr.common import NDArrays

from project.fed.utils.utils import generic_get_parameters, generic_set_parameters
from project.types.common import ClientDataloaderGen, NetGen


class Client(fl.client.NumPyClient):
    """Virtual client for ray."""

    def __init__(
        self,
        cid: int | str,
        working_dir: Path,
        net_generator: NetGen,
        dataloader_gen: ClientDataloaderGen,
    ) -> None:
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
    #     """Fit the model using the provided parameters."""
    #     self.net = self.set_parameters(parameters,  config)

    # def evaluate(
    #     self,
    #     parameters: NDArrays,
    #     config: Dict,
    # ) -> EvalRes:
    #     """Evaluate the model using the provided parameters."""
    #     self.net = self.set_parameters(parameters,  config)

    def get_parameters(self, config: Dict) -> NDArrays:
        """Implement how to get parameters."""
        if self.net is None:
            return generic_get_parameters(self.net_generator(config))
        return generic_get_parameters(self.net)

    def set_parameters(self, parameters: NDArrays, config: Dict) -> nn.Module:
        """Implement how to set parameters."""
        net = self.net_generator(config)
        generic_set_parameters(net, parameters, to_copy=False)
        return net

    def __repr__(self) -> str:
        """Implement the string representation."""
        return f"Client(cid={self.cid})"

    def get_properties(self, config: Dict) -> Dict:
        """Implement how to get properties."""
        return {}


def get_client_generator(
    working_dir: Path, net_generator: NetGen, dataloader_gen: ClientDataloaderGen
) -> Callable[[int | str], Client]:
    """Return a function which creates a new Client for a given config."""
    return lambda i: Client(
        i,
        working_dir=working_dir,
        net_generator=net_generator,
        dataloader_gen=dataloader_gen,
    )
