"""Common types shared across the project."""
from typing import Any, Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common import Metrics, NDArrays
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ClientFN returning a configurable client
ClientFN = Callable[
    [
        Any,
    ],
    fl.client.NumPyClient,
]


TransformType = Callable[[Any], torch.Tensor]

DatasetLoader = Callable[[Any], Dataset]

DatasetLoaderWithTransforms = Callable[[Any, TransformType, TransformType], Dataset]


NetGenerator = Callable[[Dict], nn.Module]

FitRes = Tuple[NDArrays, int, Dict]

TrainFunc = Callable[[nn.Module, DataLoader, Dict], Tuple[int, Dict]]


EvalRes = Tuple[float, int, Dict]
TestFunc = Callable[[nn.Module, DataLoader, Dict], EvalRes]

MetricsAggregationFn = Callable[[List[Tuple[int, Metrics]], Any], Metrics]
