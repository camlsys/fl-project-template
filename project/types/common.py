"""Typing shared across the project."""
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
import torch.nn as nn
from flwr.common import NDArrays
from torch.utils.data import DataLoader

NetGen = Callable[[Dict], nn.Module]

ClientDataloaderGen = Callable[[str | int, bool, Dict], DataLoader]
FedDataloaderGen = Callable[[bool, Dict], DataLoader]

ClientGen = Callable[[int | str], fl.client.NumPyClient]
FitRes = Tuple[NDArrays, int, Dict]
EvalRes = Tuple[float, int, Dict]

FedEvalFN = Callable[
    [int, NDArrays, Dict],
    Optional[Tuple[float, Dict]],
]

OnFitConfigFN = Callable[[int], Dict]
OnEvaluateConfigFN = OnFitConfigFN
