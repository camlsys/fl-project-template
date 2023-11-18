"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, cast

import flwr as fl
import hydra
from flwr.common.logger import log
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb
from project.client.mnist_client import (
    get_client_generator as get_client_generator_mnist,
)
from project.fed.server.deterministic_client_manager import DeterministicClientManager

# Only import from the project root
# Never do a relative import nor one that assumes a given folder structure
from project.fed.server.wandb_history import WandbHistory
from project.fed.server.wandb_server import WandbServer
from project.fed.utils.utils import (
    get_initial_parameters,
    get_save_parameters_to_file,
    get_weighted_avg_metrics_agg_fn,
    test_client,
)
from project.task.mnist_classification.dataset import get_dataloader_generators
from project.task.mnist_classification.models import get_net as get_net_mnist
from project.task.mnist_classification.train_test import (
    get_fed_eval_fn as get_fed_eval_fn_mnist,
)
from project.task.mnist_classification.train_test import (
    get_on_evaluate_config_fn as get_on_evaluate_config_fn_mnist,
)
from project.task.mnist_classification.train_test import (
    get_on_fit_config_fn as get_on_fit_config_fn_mnist,
)
from project.types.common import (
    ClientGen,
    FedEvalFN,
    NetGen,
    OnEvaluateConfigFN,
    OnFitConfigFN,
)
from project.utils.utils import (
    FileSystemManager,
    RayContextManager,
    seed_everything,
    wandb_init,
)

# Make debugging easier when using Hydra + Ray
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["OC_CAUSE"] = "1"


@hydra.main(config_path="conf", config_name="mnist", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Obtain the output dir from hydra
    original_hydra_dir = Path(
        hydra.utils.to_absolute_path(HydraConfig.get().runtime.output_dir)
    )

    output_directory = original_hydra_dir

    # Reuse an output directory for checkpointing
    if cfg.reuse_output_dir is not None:
        output_directory = Path(cfg.reuse_output_dir)

    # The directory to save data to
    results_dir = output_directory / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Where to save files to and from
    if cfg.working_dir is not None:
        # Pre-defined directory
        working_dir = Path(cfg.working_dir)
    else:
        # Default directory
        working_dir = output_directory / "working"

    working_dir.mkdir(parents=True, exist_ok=True)

    # Wandb context manager
    # controlls if wandb is initialised or not
    # if not it returns a dummy run
    with wandb_init(
        cfg.use_wandb,
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),
        config=wandb_config,  # type: ignore
    ) as run:
        log(logging.INFO, "Wandb run initialized with %s", cfg.use_wandb)

        # Context managers for saving and cleaning up files
        # from working directory at start/end of simulation
        # The RayContextManager delets the ray session folder
        with FileSystemManager(
            working_dir=working_dir,
            output_dir=results_dir,
            to_clean_once=cfg.to_clean_once,
            to_save_once=cfg.to_save_once,
            original_hydra_dir=original_hydra_dir,
            reuse_output_dir=cfg.reuse_output_dir,
            file_limit=cfg.file_limit,
        ) as fs_manager, RayContextManager() as _ray_manager:
            # Which files to save every <to_save_per_round> rounds
            # e.g model checkpoints
            save_files_per_round = fs_manager.get_save_files_every_round(
                cfg.to_save_per_round,
                cfg.save_frequency,
            )

            # For checkpointed runs, adjust the seed
            # so different clients are sampled
            adjusted_seed = cfg.fed.seed + fs_manager.checkpoint_index

            save_parameters_to_file = get_save_parameters_to_file(working_dir)

            client_manager = DeterministicClientManager(
                adjusted_seed, cfg.fed.enable_resampling
            )
            history = WandbHistory(cfg.use_wandb)

            # Keep this style if you want to dynamically
            # choose the functions using the Hydra config
            net_generator: NetGen = get_net_mnist
            get_client_dataloader, get_federated_dataloader = get_dataloader_generators(
                Path(cfg.dataset.partition_dir)
            )
            evaluate_fn: Optional[FedEvalFN] = get_fed_eval_fn_mnist(
                net_generator, get_federated_dataloader(True, cfg.fed.fed_test_config)
            )

            on_fit_config_fn: Optional[OnFitConfigFN] = get_on_fit_config_fn_mnist(
                cast(dict, OmegaConf.to_container(cfg.client.fit_config))
            )
            on_evaluate_config_fn: Optional[
                OnEvaluateConfigFN
            ] = get_on_evaluate_config_fn_mnist(
                cast(dict, OmegaConf.to_container(cfg.client.eval_config))
            )

            if cfg.fed.load_saved_parameters:
                parameters_path = (
                    results_dir / "parameters"
                    if cfg.fed.use_results_dir
                    else Path(cfg.fed.parameters_folder)
                )
            else:
                parameters_path = None

            initial_parameters = get_initial_parameters(
                net_generator,
                cast(dict, OmegaConf.to_container(cfg.fed.initial_parameters_config)),
                parameters_path,
                cfg.fed.parameters_round,
            )

            # 4. Define your strategy
            # pass all relevant argument
            # Fraction_fit and fraction_evaluate are ignored
            # in favour of using absolute numbers via min_fit_clients
            strategy = instantiate(
                cfg.strategy.init,
                fraction_fit=sys.float_info.min,
                fraction_evaluate=sys.float_info.min,
                min_fit_clients=cfg.fed.num_clients_per_round,
                min_evaluate_clients=cfg.fed.num_evaluate_clients_per_round,
                min_available_clients=cfg.fed.num_total_clients,
                on_fit_config_fn=on_fit_config_fn,
                on_evaluate_config_fn=on_evaluate_config_fn,
                evaluate_fn=evaluate_fn,
                accept_failures=False,
                fit_metrics_aggregation_fn=get_weighted_avg_metrics_agg_fn(
                    cfg.client.fit_metrics
                ),
                evaluate_metrics_aggregation_fn=get_weighted_avg_metrics_agg_fn(
                    cfg.client.evaluate_metrics
                ),
                initial_parameters=initial_parameters,
            )

            server = WandbServer(
                client_manager=client_manager,
                history=history,
                strategy=strategy,
                save_parameters_to_file=save_parameters_to_file,
                save_files_per_round=save_files_per_round,
            )

            client_generator: ClientGen = get_client_generator_mnist(
                working_dir=working_dir,
                net_generator=net_generator,
                dataloader_gen=get_client_dataloader,
            )
            seed_everything(adjusted_seed)

            test_client(
                test_all_clients=cfg.test_clients.all,
                test_one_client=cfg.test_clients.one,
                client_generator=client_generator,
                initial_parameters=initial_parameters,
                total_clients=cfg.fed.num_total_clients,
                on_fit_config_fn=on_fit_config_fn,
                on_evaluate_config_fn=on_evaluate_config_fn,
            )

            # 5. Start Simulation

            fl.simulation.start_simulation(
                client_fn=client_generator,
                num_clients=cfg.fed.num_total_clients,
                client_resources={
                    "num_cpus": int(cfg.fed.cpus_per_client),
                    "num_gpus": int(cfg.fed.gpus_per_client),
                },
                server=server,
                config=fl.server.ServerConfig(num_rounds=cfg.fed.num_rounds),
                ray_init_args={
                    "include_dashboard": False,
                    "address": cfg.ray_address,
                    "_redis_password": cfg.ray_redis_password,
                    "_node_ip_address": cfg.ray_node_ip_address,
                }
                if cfg.ray_address is not None
                else {"include_dashboard": False},
            )

            histories_dir = working_dir / "histories"
            histories_dir.mkdir(parents=True, exist_ok=True)

            with open(histories_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history.__dict__, f, ensure_ascii=False)

        if run is not None:
            run.save(
                str((results_dir / "*").resolve()),
                str((results_dir).resolve()),
                "now",
            )
            log(
                logging.INFO,
                subprocess.run(
                    ["wandb", "sync", "--clean-old-hours", "24"],
                    capture_output=True,
                    text=True,
                    check=True,
                ),
            )
    log(logging.INFO, f"{cfg.reuse_output_dir} {original_hydra_dir}")


if __name__ == "__main__":
    out_main = main()
