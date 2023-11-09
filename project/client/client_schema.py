"""The default client implementation."""
import abc
from typing import Dict, Tuple

import flwr as fl
from flwr.common import NDArrays
from pydantic import BaseModel


class ClientConfiguration(BaseModel):
    """Pydantic schema for run configuration."""

    NetGeneratorConfig: dict
    DatasetConfig: dict
    RunConfig: dict

    class Config:
        """Allow arbirary types."""

        arbitrary_types_allowed = True


class ConfigurableClient(fl.client.NumPyClient, abc.ABC):
    """Abstract class for recursive clients with configurable schemas."""

    @abc.abstractmethod
    def fit(
        self, parameters: NDArrays, config: Dict | ClientConfiguration
    ) -> Tuple[NDArrays, int, Dict]:
        """Execute the node subtree and fit the node model."""

    @abc.abstractmethod
    def evaluate(
        self, parameters: NDArrays, config: Dict | ClientConfiguration
    ) -> Tuple[float, int, Dict]:
        """Execute the node subtree and evaluate the node model."""
