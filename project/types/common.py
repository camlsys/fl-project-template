"""Typing shared across the project meant to define a stable API.

Prefer these interfaces over ad-hoc inline definitions or concrete types.
"""

from collections.abc import Callable
from pathlib import Path

import flwr as fl
from flwr.common import NDArrays
from torch import nn
from torch.utils.data import DataLoader

# Interface for network generators
# all behaviour mutations should be done
# via closures or the config
NetGen = Callable[[dict], nn.Module]

# Dataloader generators for clients and server

# Client dataloaders require the client id,
# weather the dataloader is for training or evaluation
# and the config
ClientDataloaderGen = Callable[
    [str | int, bool, dict],
    DataLoader,
]

# Server dataloaders only require a config and
# weather the dataloader is for training or evaluation
FedDataloaderGen = Callable[[bool, dict], DataLoader]

# Client generators require the client id only
# necessary for ray instantiation
# all changes in behaviour should be done via a closure
ClientGen = Callable[[int | str], fl.client.NumPyClient]

TrainFunc = Callable[
    [nn.Module, DataLoader, dict, Path],
    tuple[int, dict],
]
TestFunc = Callable[
    [nn.Module, DataLoader, dict, Path],
    tuple[float, int, dict],
]

# Type aliases for fit and eval results
# discounting the Dict[str,Scalar] typing
# of the original flwr types
FitRes = tuple[NDArrays, int, dict]
EvalRes = tuple[float, int, dict]

# A federated evaluation function
# used by the server to test the model between rounds
# requires the round number, the model parameters
# and the config
# returns the test loss and the metrics
FedEvalFN = Callable[
    [int, NDArrays, dict],
    tuple[float, dict] | None,
]

FedEvalGen = Callable[
    [NetGen, FedDataloaderGen, TestFunc, dict, Path],
    FedEvalFN | None,
]

# Functions to generate config dictionaries
# for fit and evaluate
OnFitConfigFN = Callable[[int], dict]
OnEvaluateConfigFN = OnFitConfigFN

# Structures to define a complete task setup
# They can be varied indendently to some extent
# Allows us to take advantage of hydra without
# losing static type checking
TrainStructure = tuple[TrainFunc, TestFunc, FedEvalGen]
DataStructure = tuple[
    NetGen,
    ClientDataloaderGen,
    FedDataloaderGen,
]
ConfigStructure = tuple[OnFitConfigFN, OnEvaluateConfigFN]
