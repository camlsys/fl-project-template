"""FL-related utility functions for the project."""

import logging
import struct
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch as torch
import torch.nn as nn
from flwr.common import (
    NDArrays,
    Parameters,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from project.types.common import ClientGen, NetGen, OnEvaluateConfigFN, OnFitConfigFN


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


def generic_get_parameters(net: nn.Module) -> NDArrays:
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
    byte_data = []
    if path.suffix == ".bin":
        with open(path, "rb") as f:
            while True:
                # Read the length (4 bytes)
                length_bytes = f.read(4)
                if not length_bytes:
                    break  # End of file
                length = struct.unpack("I", length_bytes)[0]

                # Read the data of the specified length
                data = f.read(length)
                byte_data.append(data)

        return Parameters(tensors=byte_data, tensor_type="numpy.ndarray")

    raise ValueError(f"Unknown parameter format: {path}")


def get_initial_parameters(
    net_generator: NetGen, config: Dict, load_from: Optional[Path], round=Optional[int]
) -> Parameters:
    """Get the initial parameters for the network.

    Parameters
    ----------
    net_generator : NetGen
        The function to generate the network.
    config : Dict
        The configuration.
    load_from : Optional[Path]
        The path to the parameters file.

    Returns
    -------
    'Parameters
        The parameters.
    """
    if load_from is None:
        log(logging.INFO, "Generating initial parameters with config: %s", config)
        return ndarrays_to_parameters(generic_get_parameters(net_generator(config)))
    try:
        if round is not None:
            # Load specific round parameters
            load_from = load_from / f"parameters_{round}.bin"
        else:
            # Load only the most recent parameters
            load_from = max(
                Path(load_from).glob("parameters_*.bin"),
                key=lambda f: (int(f.stem.split("_")[1]), int(f.stem.split("_")[2])),
            )

        log(logging.INFO, "Loading initial parameters from: %s", load_from)

        return load_parameters_from_file(load_from)
    except (
        ValueError,
        FileNotFoundError,
        PermissionError,
        OSError,
        EOFError,
        IsADirectoryError,
    ):
        log(logging.INFO, f"Loading parameters failed from: {load_from}")
        log(logging.INFO, "Generating initial parameters with config: %s", config)

        return ndarrays_to_parameters(generic_get_parameters(net_generator(config)))


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
        """Save the parameters to a file.

        Parameters
        ----------
        parameters : Parameters
            The parameters to save.

        Returns
        -------
        None
        """
        parameters_path = working_dir / "parameters"
        parameters_path.mkdir(parents=True, exist_ok=True)
        with open(parameters_path / "parameters.bin", "wb") as f:
            # Since Parameters is a list of bytes
            # save the length of each row and the data
            # for deserialization
            for data in parameters.tensors:
                # Prepend the length of the data as a 4-byte integer
                f.write(struct.pack("I", len(data)))
                f.write(data)

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


def test_client(
    test_all_clients: bool,
    test_one_client: bool,
    client_generator: ClientGen,
    initial_parameters: Parameters,
    total_clients: int,
    on_fit_config_fn: Optional[OnFitConfigFN],
    on_evaluate_config_fn: Optional[OnEvaluateConfigFN],
) -> None:
    """Debug the client code.

    Avoids the complexity of Ray.
    """
    parameters = parameters_to_ndarrays(initial_parameters)
    if test_all_clients or test_one_client:
        if test_one_client:
            client = client_generator(0)
            _, *res_fit = client.fit(
                parameters, on_fit_config_fn(0) if on_fit_config_fn else {}
            )
            res_eval = client.evaluate(
                parameters, on_evaluate_config_fn(0) if on_evaluate_config_fn else {}
            )
            log(logging.INFO, "Fit debug fit: %s  and eval: %s", res_fit, res_eval)
        else:
            for i in range(total_clients):
                client = client_generator(i)
                _, *res_fit = client.fit(
                    parameters, on_fit_config_fn(i) if on_fit_config_fn else {}
                )
                res_eval = client.evaluate(
                    parameters,
                    on_evaluate_config_fn(i) if on_evaluate_config_fn else {},
                )
                log(logging.INFO, "Fit debug fit: %s  and eval: %s", res_fit, res_eval)
