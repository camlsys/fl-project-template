"""FL-related utility functions for the project."""

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, SupportsIndex, Tuple, cast

import numpy as np
import torch as torch
import torch.nn as nn
from flwr.common import NDArrays, Parameters


def generic_set_parameters(net: nn.Module, parameters: NDArrays, to_copy=False) -> None:
    """Set the parameters of a network."""
    params_dict = zip(net.state_dict().keys(), parameters, strict=True)
    state_dict = OrderedDict(
        {k: torch.Tensor(v if not to_copy else v.copy()) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)


def generic_get_parameters(net: torch.nn.Module) -> NDArrays:
    """Implement generic `get_parameters` for Flower Client."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def load_ndarrays_from_file(path: Path) -> NDArrays:
    """Load parameters from a file."""
    if path.suffix == ".npy" or path.suffix == ".npz" or path.suffix == ".np":
        return list(np.load(file=str(path), allow_pickle=True).values())
    if path.suffix == ".pt":
        return torch.load(path)

    raise ValueError(f"Unknown parameter format: {path}")


def load_parameters_from_file(path: Path) -> Parameters:
    """Load parameters from a file."""
    if path.suffix == ".bin":
        return Parameters(
            tensors=cast(List[bytes], list(path.read_bytes())), tensor_type="str"
        )

    raise ValueError(f"Unknown parameter format: {path}")


def get_save_parameters_to_file(working_dir: Path) -> Callable[[Parameters], None]:
    """Get a function to save parameters to a file."""

    def save_parameters_to_file(parameters: Parameters) -> None:
        """Save the parameters to a file."""
        parameters_path = working_dir / "parameters"
        parameters_path.mkdir(parents=True, exist_ok=True)
        with open(parameters_path / "parameters.bin", "wb") as f:
            f.write(bytearray(cast(Iterable[SupportsIndex], parameters.tensors)))

    return save_parameters_to_file


def get_weighted_avg_metrics_agg_fn(
    to_agg: Set[str],
) -> Callable[[List[Tuple[int, Dict]]], Dict]:
    """Return a function to compute a weighted average over pre-defined metrics."""

    def weighted_avg(metrics: List[Tuple[int, Dict]]) -> Dict:
        """Compute a weighted average over pre-defined metrics."""
        total_num_examples = sum([num_examples for num_examples, _ in metrics])
        weighted_metrics: Dict = defaultdict(float)
        for num_examples, metric in metrics:
            for key, value in metric.items():
                if key in to_agg:
                    weighted_metrics[key] += num_examples * value

        return {
            key: value / total_num_examples for key, value in weighted_metrics.items()
        }

    return weighted_avg
