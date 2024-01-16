"""FL-related utility functions for the project."""

import json
import random
import logging
import struct
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
import numpy as np

import torch
from flwr.common import (
    NDArrays,
    Parameters,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.history import History
from torch import nn
from project.fed.server.wandb_history import WandbHistory

from project.types.common import (
    ClientCIDandSeedGeneratorsState,
    ClientGen,
    Ext,
    Files,
    Folders,
    IsolatedRNGState,
    IsolatedRNG,
    ServerRNG,
    NetGen,
    OnEvaluateConfigFN,
    OnFitConfigFN,
    RNGStateTuple,
)
from project.utils.utils import obtain_device


def generic_set_parameters(
    net: nn.Module,
    parameters: NDArrays,
    to_copy: bool = True,
) -> None:
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
    params_dict = zip(
        net.state_dict().keys(),
        parameters,
        strict=True,
    )
    state_dict = OrderedDict(
        {k: torch.Tensor(v if not to_copy else v.copy()) for k, v in params_dict},
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
    if path.suffix == f".{Ext.PARAMETERS}":
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
        log(logging.INFO, f"Loaded parameters from: {path}")
        return Parameters(
            tensors=byte_data,
            tensor_type="numpy.ndarray",
        )

    raise ValueError(f"Unknown parameter format: {path}")


def get_state(
    net_generator: NetGen,
    config: dict,
    load_parameters_from: Path | None,
    load_rng_from: Path | None,
    load_history_from: Path | None,
    seed: int,
    server_round: int,
    use_wandb: bool,
) -> tuple[Parameters, ServerRNG, History]:
    """Get initial parameters for the network+rng state if starting from a checkpoint.

    Parameters
    ----------
    net_generator : NetGen
        The function to generate the network.
    config : Dict
        The configuration.
    load_from : Optional[Path]
        The path to the state folder.

    Returns
    -------
    'Parameters
        The parameters.
    """
    parameters_path = None
    if load_parameters_from is None:
        log(
            logging.INFO,
            "Generating initial parameters with config: %s",
            config,
        )
        server_rng_tuple = load_and_set_rng(seed, None, None)

        return (
            ndarrays_to_parameters(
                generic_get_parameters(net_generator(config, server_rng_tuple[0])),
            ),
            server_rng_tuple,
            load_history(None, None, use_wandb),
        )
    try:
        if server_round is None:
            return get_state(
                net_generator,
                config,
                None,
                load_rng_from=None,
                load_history_from=None,
                seed=seed,
                server_round=server_round,
                use_wandb=use_wandb,
            )
        parameters_path = (
            load_parameters_from / f"{Files.PARAMETERS}_{server_round}.{Ext.PARAMETERS}"
        )
        return (
            load_parameters_from_file(parameters_path),
            load_and_set_rng(seed, load_rng_from, server_round),
            load_history(load_history_from, server_round, use_wandb),
        )
    except (
        ValueError,
        FileNotFoundError,
        PermissionError,
        OSError,
        EOFError,
        IsADirectoryError,
    ):
        log(
            logging.INFO,
            f"""Loading parameters failed from: {parameters_path}""",
        )
        log(
            logging.INFO,
            "Generating initial parameters with config: %s",
            config,
        )
        # raise

        return get_state(
            net_generator,
            config,
            None,
            load_rng_from=None,
            load_history_from=None,
            seed=seed,
            server_round=server_round,
            use_wandb=use_wandb,
        )


def get_save_parameters_to_file(
    working_dir: Path,
) -> Callable[[Parameters], None]:
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

    def save_parameters_to_file(
        parameters: Parameters,
    ) -> None:
        """Save the parameters to a file.

        Parameters
        ----------
        parameters : Parameters
            The parameters to save.

        Returns
        -------
        None
        """
        parameters_path = working_dir / Folders.STATE / Folders.PARAMETERS
        parameters_path.mkdir(parents=True, exist_ok=True)
        with open(
            parameters_path / f"{Files.PARAMETERS}.{Ext.PARAMETERS}",
            "wb",
        ) as f:
            # Since Parameters is a list of bytes
            # save the length of each row and the data
            # for deserialization
            for data in parameters.tensors:
                # Prepend the length of the data as a 4-byte integer
                f.write(struct.pack("I", len(data)))
                f.write(data)

    return save_parameters_to_file


def get_save_history_to_file(
    working_dir: Path,
) -> Callable[[History], None]:
    """Get a function to save history to a file.

    Parameters
    ----------
    working_dir : Path
        The working directory.

    Returns
    -------
    Callable[[History], None]
        A function to save history to a file.
    """

    def save_history_to_file(
        history: History,
    ) -> None:
        """Save the history to a file.

        Parameters
        ----------
        history : History
            The history to save.

        Returns
        -------
        None
        """
        history_path = working_dir / Folders.STATE / Folders.HISTORIES
        history_path.mkdir(parents=True, exist_ok=True)
        with open(
            history_path / f"{Files.HISTORY}.{Ext.HISTORY}",
            "w",
            encoding="utf-8",
        ) as f:
            # Since Parameters is a list of bytes
            # save the length of each row and the data
            # for deserialization
            json.dump(
                history.__dict__,
                f,
                ensure_ascii=False,
            )

    return save_history_to_file


def extract_rng_state(
    rng: random.Random | np.random.Generator | torch.Generator | None,
) -> tuple[Any, ...] | dict[str, Any] | torch.Tensor | None:
    """Get the state of a random number generator.

    Parameters
    ----------
    rng : random.Random | np.random.Generator | torch.Generator
        The random number generator.

    Returns
    -------
    Tuple[Any, ...] | Dict[str, Any] | torch.Tensor
        The state of the random number generator.
    """
    if rng is None:
        return None

    if isinstance(rng, random.Random):
        return rng.getstate()
    if isinstance(rng, np.random.Generator):
        return rng.__getstate__()
    if isinstance(rng, torch.Generator):
        return rng.get_state()
    raise ValueError(f"Unknown random number generator: {rng}")


def get_save_rng_to_file(
    working_dir: Path,
) -> Callable[[ServerRNG], None]:
    """Get a function to save the rng state to a file.

    Parameters
    ----------
    working_dir : Path
        The working directory.

    Returns
    -------
    Callable[[ServerRNG], None]
        A function to save the rng state to a file.
    """

    def save_rng_to_file(
        sever_rng: ServerRNG,
    ) -> None:
        """Save the rng state to a file.

        Parameters
        ----------
        sever_rng : ServerRNG
            The rng state to save.

        Returns
        -------
        None
        """
        rng_path = working_dir / Folders.STATE / Folders.RNG
        rng_path.mkdir(parents=True, exist_ok=True)

        rng_global_state = random.getstate()
        np_rng_global_state = np.random.get_state()
        torch_rng_global_state = torch.get_rng_state()
        (seed, *rng_tuple), *cid_seed_generators = sever_rng

        rng_tuple_state: IsolatedRNGState = cast(
            IsolatedRNGState,
            tuple([seed] + [extract_rng_state(rng) for rng in rng_tuple]),
        )

        cid_seed_generators_state: ClientCIDandSeedGeneratorsState = cast(
            ClientCIDandSeedGeneratorsState,
            tuple(extract_rng_state(rng) for rng in cid_seed_generators),
        )
        state: RNGStateTuple = (
            (rng_global_state, np_rng_global_state, torch_rng_global_state),
            rng_tuple_state,
            cid_seed_generators_state,
        )

        torch.save(state, rng_path / f"{Files.RNG_STATE}.{Ext.RNG_STATE}")

    return save_rng_to_file


def get_weighted_avg_metrics_agg_fn(
    to_agg: set[str],
) -> Callable[[list[tuple[int, dict]]], dict]:
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

    def weighted_avg(
        metrics: list[tuple[int, dict]],
    ) -> dict:
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
        total_num_examples = sum(
            [num_examples for num_examples, _ in metrics],
        )
        weighted_metrics: dict = defaultdict(float)
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
    on_fit_config_fn: OnFitConfigFN | None,
    on_evaluate_config_fn: OnEvaluateConfigFN | None,
) -> None:
    """Debug the client code.

    Avoids the complexity of Ray.
    """
    parameters = parameters_to_ndarrays(initial_parameters)
    if test_all_clients or test_one_client:
        if test_one_client:
            client = client_generator(str(0))
            _, *res_fit = client.fit(
                parameters,
                on_fit_config_fn(0) if on_fit_config_fn else {},
            )
            res_eval = client.evaluate(
                parameters,
                on_evaluate_config_fn(0) if on_evaluate_config_fn else {},
            )
            log(
                logging.INFO,
                "Fit debug fit: %s  and eval: %s",
                res_fit,
                res_eval,
            )
        else:
            for i in range(total_clients):
                client = client_generator(str(i))
                _, *res_fit = client.fit(
                    parameters,
                    on_fit_config_fn(i) if on_fit_config_fn else {},
                )
                res_eval = client.evaluate(
                    parameters,
                    on_evaluate_config_fn(i) if on_evaluate_config_fn else {},
                )
                log(
                    logging.INFO,
                    "Fit debug fit: %s  and eval: %s",
                    res_fit,
                    res_eval,
                )


def get_isolated_rng_tuple(seed: int, device: torch.device) -> IsolatedRNG:
    """Get the random state for server/clients.

    Parameters
    ----------
    seed : int
        The seed.
    device : torch.device
        The device.

    Returns
    -------
    Tuple[random.Random, np.random.Generator,
          torch.Generator(CPU), Optional[torch.Generator(GPU)]]
        The random state for clients.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    torch_rng_cpu = torch.Generator()

    torch_rng_cpu.manual_seed(seed)
    torch_rng_gpu = torch.Generator(device=device) if device != "cpu" else None
    if torch_rng_gpu is not None:
        torch_rng_gpu.manual_seed(seed)

    return seed, rng, np_rng, torch_rng_cpu, torch_rng_gpu


def get_isolated_rng_from_state(
    rng_state: IsolatedRNGState,
) -> IsolatedRNG:
    """Get the random state for server/clients.

    Parameters
    ----------
    rng_state : IsolatedRNGState
        The random state.

    Returns
    -------
    Tuple[random.Random, np.random.Generator,
          torch.Generator(CPU), Optional[torch.Generator(GPU)]]
        The random state for server/clients.
    """
    (
        seed,
        random_state,
        np_rng_state,
        torch_rng_cpu_state,
        torch_rng_gpu_state,
    ) = rng_state
    rng = random.Random(seed)
    rng.setstate(random_state)
    np_rng = np.random.default_rng(seed)
    np_rng.__setstate__(np_rng_state)
    torch_rng_cpu = torch.Generator().set_state(torch_rng_cpu_state)
    torch_rng_gpu = (
        torch.Generator().set_state(torch_rng_gpu_state)
        if torch_rng_gpu_state is not None
        else None
    )

    return seed, rng, np_rng, torch_rng_cpu, torch_rng_gpu


def load_and_set_rng(
    seed: int, rng_path: Path | None, server_round: int | None
) -> ServerRNG:
    """Get the random state for server/clients.

    Parameters
    ----------
    seed : int
        The seed.
    rng_path : Optional[Path]
        The path to the random state.
    server_round : Optional[int]
        The server round.

    Returns
    -------
    Tuple[random.Random, np.random.Generator,
          torch.Generator(CPU), Optional[torch.Generator(GPU)]]
        The random state for server/clients.
    """
    if rng_path is None:
        # Set global RNG
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create server RNG tuple
        server_rng_tuple = get_isolated_rng_tuple(seed, obtain_device())
        server_random = server_rng_tuple[1]
        # Create client cid and seed generators
        client_cid_generator_rng = random.Random(server_random.randint(0, 2**32 - 1))
        client_seed_generator_rng = random.Random(server_random.randint(0, 2**32 - 1))
        log(logging.INFO, f"Using RNG seed: {seed}")
        return server_rng_tuple, client_cid_generator_rng, client_seed_generator_rng

    if (
        server_round is not None
        and rng_path is not None
        and (
            rng_file_path := rng_path
            / f"{Files.RNG_STATE}_{server_round}.{Ext.RNG_STATE}"
        ).exists()
    ):
        states: RNGStateTuple = torch.load(rng_file_path)

        (
            (random_global_state, np_global_state, torch__global_state),
            server_state,
            (client_cid_generator_state, client_seed_generator_state),
        ) = states

        # Set global states
        random.setstate(random_global_state)
        np.random.set_state(np_global_state)
        torch.set_rng_state(torch__global_state)

        # Set the server rng
        server_rng_tuple = get_isolated_rng_from_state(server_state)
        saved_seed = server_rng_tuple[0]

        # Set the cid and seed generators for the clients
        client_cid_generator_rng = random.Random(saved_seed)
        client_cid_generator_rng.setstate(client_cid_generator_state)
        client_seed_generator_rng = random.Random(saved_seed)
        client_seed_generator_rng.setstate(client_seed_generator_state)

        log(logging.INFO, f"Loading RNG state from: {rng_file_path}")
        return server_rng_tuple, client_cid_generator_rng, client_seed_generator_rng

    return load_and_set_rng(seed, None, None)


def load_history(
    history_path: Path | None, server_round: int | None, use_wandb: bool
) -> History:
    """Load the history.

    Parameters
    ----------
    history_path : Optional[Path]
        The path to the history directory.
    server_round : Optional[int]
        The server round.

    Returns
    -------
    History
        The history.
    """
    if history_path is None:
        # Init history
        return WandbHistory(use_wandb)
    if (
        server_round is not None
        and history_path is not None
        and (
            history_file_path := history_path
            / f"{Files.HISTORY}_{server_round}.{Ext.HISTORY}"
        ).exists()
    ):
        history = WandbHistory(use_wandb)

        # Load previous history
        with open(history_file_path, encoding="utf-8") as f:
            history.__dict__ = json.load(f)
        return history

    return load_history(None, None, use_wandb)
