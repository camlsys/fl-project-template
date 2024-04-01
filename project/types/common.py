"""Typing shared across the project meant to define a stable API.

Prefer these interfaces over ad-hoc inline definitions or concrete types.
"""

from collections.abc import Callable
from pathlib import Path
import random
from typing import Any

import flwr as fl
from flwr.common import NDArrays
import numpy as np
from omegaconf import DictConfig
from torch import nn
import torch
from torch.utils.data import DataLoader
import enum

CID = str | int | Path

GlobalState = tuple[
    # Random
    tuple[Any, ...],
    # np
    dict[str, Any],
    # torch
    torch.Tensor,
]

IsolatedRNGState = tuple[
    # Seed
    int,
    # Random
    tuple[Any, ...],
    # np
    dict[str, Any],
    # torch
    torch.Tensor,
    # torch GPU
    torch.Tensor | None,
]

ClientCIDandSeedGeneratorsState = tuple[
    # Client cid generator
    tuple[Any, ...],
    # Client seed generator
    tuple[Any, ...],
]


# Contains the rng state for
# Global: Random, NP, Torch rng
# The RNG tuple of the server
# Client CID Generator
# Client Seed Generator
RNGStateTuple = tuple[
    # Global state
    GlobalState,
    # Server RNG state
    IsolatedRNGState,
    # Client cid and seed generators
    ClientCIDandSeedGeneratorsState,
]


# Necessary to guarantee reproducibility across all sources of randomness
IsolatedRNG = tuple[
    int, random.Random, np.random.Generator, torch.Generator, torch.Generator | None
]

# Payload for the generators controlling server behavior
ServerRNG = tuple[
    # Server RNG tuple
    IsolatedRNG,
    # Client cid and seed generators
    random.Random,
    random.Random,
]


# Interface for network generators
# all behavior mutations should be done
# via closures or the config
NetGen = Callable[
    [
        dict,
        IsolatedRNG,
        DictConfig | None,
    ],
    nn.Module,
]

# Dataloader generators for clients and server

# Client dataloaders require the client id,
# weather the dataloader is for training or evaluation
# and the config
ClientDataloaderGen = Callable[
    [
        CID,
        bool,
        dict,
        IsolatedRNG,
        DictConfig | None,
    ],
    DataLoader,
]

# Server dataloaders only require a config and
# weather the dataloader is for training or evaluation
FedDataloaderGen = Callable[
    [
        bool,
        dict,
        IsolatedRNG,
        DictConfig | None,
    ],
    DataLoader,
]

# Client generators require the client id only
# necessary for ray instantiation
# all changes in behavior should be done via a closure
ClientGen = Callable[[str], fl.client.NumPyClient]

TrainFunc = Callable[
    [
        nn.Module | NDArrays,
        DataLoader | None,
        dict,
        Path,
        IsolatedRNG,
        DictConfig | None,
    ],
    tuple[nn.Module | NDArrays, int, dict],
]
TestFunc = Callable[
    [
        nn.Module | NDArrays,
        DataLoader | None,
        dict,
        Path,
        IsolatedRNG,
        DictConfig | None,
    ],
    tuple[float, int, dict],
]

# Type aliases for fit and eval results
# discounting the Dict[str,Scalar] typing
# of the original flwr types
FitRes = tuple[NDArrays, int, dict]
EvalRes = tuple[float, int, dict]


# A function to initialize the working directory
# in case your training relies on a specific
# directory structure pre-existing
InitWorkingDir = Callable[
    [
        # The working dir path
        Path,
        # The results dir path
        Path,
    ],
    None,
]

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
    [
        NetGen | None,
        FedDataloaderGen | None,
        TestFunc,
        dict,
        Path,
        IsolatedRNG,
        DictConfig | None,
    ],
    FedEvalFN | None,
]

# Functions to generate config dictionaries
# for fit and evaluate
OnFitConfigFN = Callable[[int], dict]
OnEvaluateConfigFN = OnFitConfigFN

# Structures to define a complete task setup
# They can be varied independently to some extent
# Allows us to take advantage of hydra without
# losing static type checking
TrainStructure = tuple[TrainFunc, TestFunc, FedEvalGen]
DataStructure = tuple[
    NetGen | None,
    ClientDataloaderGen | None,
    FedDataloaderGen | None,
    InitWorkingDir | None,
]
ConfigStructure = tuple[OnFitConfigFN, OnEvaluateConfigFN]


GetClientGen = Callable[
    [
        # The working directory
        Path,
        # The network generator
        NetGen | None,
        # The client dataloader generator
        ClientDataloaderGen | None,
        # The training function
        TrainFunc,
        # The testing function
        TestFunc,
        # Seeded rng for client seed initialization
        random.Random,
        # Hydra config
        DictConfig | None,
    ],
    ClientGen,
]


class IntentionalDropoutError(Exception):
    """Exception raised when a client intentionally drops out."""


class FileCountExceededError(Exception):
    """Exception raised when a client intentionally drops out."""


class Folders(enum.StrEnum):
    """Enum for folder types."""

    WORKING = enum.auto()
    STATE = enum.auto()
    PARAMETERS = enum.auto()
    RNG = enum.auto()
    HISTORIES = enum.auto()
    HYDRA = ".hydra"
    RESULTS = enum.auto()
    WANDB = enum.auto()


class Files(enum.StrEnum):
    """Enum for file types."""

    PARAMETERS = enum.auto()
    RNG_STATE = "rng-state"
    HISTORY = enum.auto()
    MAIN = enum.auto()
    WANDB_RUN = enum.auto()


class Ext(enum.StrEnum):
    """Enum for file extensions."""

    PARAMETERS = "bin"
    RNG_STATE = "pt"
    HISTORY = "json"
    MAIN = "log"
    WANDB_RUN = "json"
