"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
# these are the basic packages you'll need here
# feel free to remove some if aren't needed
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import flwr as fl
import hydra
from flwr.common.logger import log
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb
from project.client.client import get_client_generator as get_default_client_generator
from project.fed.server.deterministic_client_manager import DeterministicClientManager

# Only import from the project root
# Never do a relative import nor one that assumes a given folder structure
from project.fed.server.wandb_history import WandbHistory
from project.fed.server.wandb_server import WandbServer
from project.fed.utils.utils import get_weighted_avg_metrics_agg_fn
from project.task.default.models import get_model as get_default_model
from project.task.default.train_test import get_fed_eval_fn as get_default_fed_eval_fn
from project.task.default.train_test import (
    get_on_evaluate_config_fn as get_default_on_evaluate_config_fn,
)
from project.task.default.train_test import (
    get_on_fit_config_fn as get_default_on_fit_config_fn,
)
from project.typing.common import (
    ClientGenerator,
    FedEvalFN,
    NetGenerator,
    OnEvaluateConfigFN,
    OnFitConfigFN,
)
from project.utils.utils import (
    FileSystemManager,
    RayContextManager,
    get_save_files_every_round,
    wandb_init,
)

# Make debugging easier when using Hydra + Ray
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["OC_CAUSE"] = "1"


@hydra.main(config_path="conf", config_name="base", version_base=None)
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
    output_directory = Path(
        hydra.utils.to_absolute_path(HydraConfig.get().runtime.output_dir)
    )

    # Reuse an output directory for checkpointing
    if cfg.reuse_output_dir is not None:
        output_directory = Path(cfg.reuse_output_dir)

    # The directory to save data to
    results_dir = output_directory / "results"

    # Where to save files to and from
    if cfg.save_from_dir is not None:
        # Pre-defined directory
        working_dir = Path(cfg.working_dir)
    else:
        # Default directory
        working_dir = output_directory / "working"

    working_dir.mkdir(parents=True, exist_ok=True)

    # 2. Prepare your dataset
    # here you should call a function in project.task.<your_task>.datasets.py
    # that returns whatever is needed to:
    # (1) ensure the server can access the dataset used to evaluate your model after
    # aggregation
    # (2) tell each client what dataset partitions they should use (e.g. a this could
    # be a location in the file system, a list of dataloader, a list of ids to extract
    # from a dataset, it's up to you)
    # <Your code here>

    with wandb_init(
        cfg.use_wandb,
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),
        config=wandb_config,  # type: ignore
    ) as run:
        print("Wandb run initialized with ", cfg.use_wandb)
        with FileSystemManager(
            working_dir, output_directory, cfg.to_clean_once, cfg.to_save_once
        ) and RayContextManager() as _, _:
            save_files_per_round = get_save_files_every_round(
                working_dir,
                results_dir,
                cfg.to_save_per_round,
                cfg.save_frequency,
            )
            client_manager = DeterministicClientManager(
                cfg.fed.seed, cfg.fed.enable_resampling
            )
            history = WandbHistory(cfg.use_wandb)

            # Keep this style if you want to dynamically
            # choose the functions using the Hydra config
            net_generator: NetGenerator = get_default_model
            evaluate_fn: Optional[FedEvalFN] = get_default_fed_eval_fn()
            on_fit_config_fn: Optional[OnFitConfigFN] = get_default_on_fit_config_fn()
            on_evaluate_config_fn: Optional[
                OnEvaluateConfigFN
            ] = get_default_on_evaluate_config_fn()

            # 4. Define your strategy
            # pass all relevant argument

            instantiate(
                cfg.strategy.init,
                fraction_fit=(
                    float(cfg.fed.num_clients_per_round) / cfg.fed.num_total_clients
                ),
                fraction_evaluate=(
                    float(cfg.fed.num_evaluate_clients_per_round)
                    / cfg.fed.num_total_clients
                ),
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
            )

            server = WandbServer(
                client_manager=client_manager,
                history=history,
                strategy=None,
                save_files_per_round=save_files_per_round,
            )

            client_generator: ClientGenerator = get_default_client_generator(
                working_dir=working_dir, net_generator=net_generator
            )

            # 5. Start Simulation
            fl.simulation.start_simulation(
                client_fn=client_generator,
                num_clients=cfg.fed.num_total_clients,
                client_resources={
                    "num_cpus": cfg.fed.cpus_per_client,
                    "num_gpus": cfg.fed.gpus_per_client,
                },
                server=server,
                config=fl.server.ServerConfig(num_rounds=cfg.fed.num_rounds),
                ray_init_args={"include_dashboard": False},
            )

            histories_dir = working_dir / "histories"
            histories_dir.mkdir(parents=True, exist_ok=True)

            with open(histories_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history.__dict__, f, ensure_ascii=False)

        if run is not None:
            run.save(
                str((output_directory / "*").resolve()),
                str((output_directory).resolve()),
                "now",
            )
            print(
                subprocess.run(
                    ["wandb", "sync", "--clean-old-hours", "24"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            )
