"""Typing shared across the project meant to define a stable API.

Prefer these interfaces over ad-hoc inline definitions or concrete types.
"""
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
import torch.nn as nn
from flwr.common import NDArrays
from torch.utils.data import DataLoader

# Interface for network generators
# all behaviour mutations should be done
# via closures or the config
NetGen = Callable[[Dict], nn.Module]

# Dataloader generators for clients and server

# Client dataloaders require the client id,
# weather the dataloader is for training or evaluation
# and the config
ClientDataloaderGen = Callable[[str | int, bool, Dict], DataLoader]

# Server dataloaders only require a config and
# weather the dataloader is for training or evaluation
FedDataloaderGen = Callable[[bool, Dict], DataLoader]

# Client generators require the client id only
# necessary for ray instantiation
# all changes in behaviour should be done via a closure
ClientGen = Callable[[int | str], fl.client.NumPyClient]

TrainFunc = Callable[[nn.Module, DataLoader, Dict, Path], Tuple[int, Dict]]
TestFunc = Callable[[nn.Module, DataLoader, Dict, Path], Tuple[float, int, Dict]]

# Type aliases for fit and eval results
# discounting the Dict[str,Scalar] typing
# of the original flwr types
FitRes = Tuple[NDArrays, int, Dict]
EvalRes = Tuple[float, int, Dict]

# A federated evaluation function
# used by the server to test the model between rounds
# requires the round number, the model parameters
# and the config
# returns the test loss and the metrics
FedEvalFN = Callable[
    [int, NDArrays, Dict],
    Optional[Tuple[float, Dict]],
]

FedEvalGen = Callable[
    [NetGen, FedDataloaderGen, TestFunc, Dict, Path], Optional[FedEvalFN]
]

# Functions to generate config dictionaries
# for fit and evaluate
OnFitConfigFN = Callable[[int], Dict]
OnEvaluateConfigFN = OnFitConfigFN

# Structures to define a complete task setup
# They can be varied indendently to some extent
# Allows us to take advantage of hydra without
# losing static type checking
TrainStructure = Tuple[TrainFunc, TestFunc, FedEvalGen]
DataStructure = Tuple[NetGen, ClientDataloaderGen, FedDataloaderGen]
ConfigStructure = Tuple[OnFitConfigFN, OnEvaluateConfigFN]
