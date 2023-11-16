"""FL-related utility functions for the project."""

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, SupportsIndex, Tuple, cast

import torch as torch
import torch.nn as nn
from flwr.common import NDArrays, Parameters


def generic_set_parameters(net: nn.Module, parameters: NDArrays, to_copy=True) -> None:
    """Set the parameters of a network.

    Parameters
    ----------
    net : nn.Module
        The network whose parameters should be set.
        parameters : NDArrays
        The parameters to set.
        to_copy : bool (default=False)
        Whether to copy the parameters or use them directly.

    Returns
    -------
        None
    """
    params_dict = zip(net.state_dict().keys(), parameters, strict=True)
    state_dict = OrderedDict(
        {k: torch.Tensor(v if not to_copy else v.copy()) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)


def generic_get_parameters(net: torch.nn.Module) -> NDArrays:
    """Implement generic `get_parameters` for Flower Client.

    Parameters
    ----------
    net : nn.Module
        The network whose parameters should be returned.

    Returns
    -------
        NDArrays
        The parameters of the network.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def load_parameters_from_file(path: Path) -> Parameters:
    """Load parameters from a binary file.

    Parameters
    ----------
    path : Path
        The path to the parameters file.

    Returns
    -------
        'Parameters
        The parameters.
    """
    if path.suffix == ".bin":
        return Parameters(
            tensors=cast(List[bytes], list(path.read_bytes())), tensor_type="str"
        )

    raise ValueError(f"Unknown parameter format: {path}")


def get_save_parameters_to_file(working_dir: Path) -> Callable[[Parameters], None]:
    """Get a function to save parameters to a file.

    Parameters
    ----------
    working_dir : Path
        The working directory.

    Returns
    -------
        Callable[[Parameters], None]
        A function to save parameters to a file.
    """

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
    """Return a function to compute a weighted average over pre-defined metrics.

    Parameters
    ----------
    to_agg : Set[str]
        The metrics to aggregate.

    Returns
    -------
        Callable[[List[Tuple[int, Dict]]], Dict]
        A function to compute a weighted average over pre-defined metrics.
    """

    def weighted_avg(metrics: List[Tuple[int, Dict]]) -> Dict:
        """Compute a weighted average over pre-defined metrics.

        Parameters
        ----------
        metrics : List[Tuple[int, Dict]]
            The metrics to aggregate.

        Returns
        -------
            Dict
            The weighted average over pre-defined metrics.
        """
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
