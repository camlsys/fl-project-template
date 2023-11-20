"""Create and connect the building blocks for your experiments; start the simulation.

It includes processing the dataset, instantiate strategy, specifying how the global
model will be evaluated, etc. In the end, this script saves the results.
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, cast

import flwr as fl
import hydra
from flwr.common.logger import log
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb

# Only import from the project root
# Never do a relative import nor one that assumes a given folder structure
from project.client.client import get_client_generator
from project.dispatch.dispatch import dispatch_config, dispatch_data, dispatch_train
from project.fed.server.deterministic_client_manager import DeterministicClientManager
from project.fed.server.wandb_history import WandbHistory
from project.fed.server.wandb_server import WandbServer
from project.fed.utils.utils import (
    get_initial_parameters,
    get_save_parameters_to_file,
    get_weighted_avg_metrics_agg_fn,
    test_client,
)
from project.types.common import ClientGen, FedEvalFN
from project.utils.utils import (
    FileSystemManager,
    RayContextManager,
    seed_everything,
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
    # Print parsed config
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
        # from the working directory
        # at the start/end of the simulation
        # The RayContextManager deletes the ray session folder
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
            # e.g. model checkpoints
            save_files_per_round = fs_manager.get_save_files_every_round(
                cfg.to_save_per_round,
                cfg.save_frequency,
            )

            # For checkpointed runs, adjust the seed
            # so different clients are sampled
            adjusted_seed = cfg.fed.seed ^ fs_manager.checkpoint_index

            save_parameters_to_file = get_save_parameters_to_file(working_dir)

            # Client manager that samples the same clients
            # For a given seed+checkpoint combination
            client_manager = DeterministicClientManager(
                adjusted_seed, cfg.fed.enable_resampling
            )

            # New history that sends data to the wandb server
            # only if use_wandb is True
            # Minimizes communication to oncer-per-round
            history = WandbHistory(cfg.use_wandb)

            # All of these functions are determined by the cfg.task component
            # change model_and_data and train_structure
            # If you want to change them
            # add functionality to project.dispatch and then
            # to the individual dispatch.py file of each task

            # Obtain the net generator, dataloader and fed_dataloader
            # Change the cfg.task.model_and_data str to change functionality
            net_generator, client_dataloader_gen, fed_dataloater_gen = dispatch_data(
                cfg
            )

            # Obtain the train/test func and the fed eval func
            # Change the cfg.task.train_structure str to change functionality
            train_func, test_func, get_fed_eval_fn = dispatch_train(cfg)

            # Obtain the on_fit config and on_eval config
            # generation functions
            # These depend on the cfg.task.fit_config
            # and cfg.task.eval_config dictionaries by default
            on_fit_config_fn, on_evaluate_config_fn = dispatch_config(cfg)

            # Build the evaluate function from the given components
            # This is the function that is called on the server
            # to evaluated the global model
            # the cast to Dict is necessary for mypy
            # as is the to_container
            evaluate_fn: Optional[FedEvalFN] = get_fed_eval_fn(
                net_generator,
                fed_dataloater_gen,
                test_func,
                cast(Dict, OmegaConf.to_container(cfg.task.fed_test_config)),
                working_dir,
            )

            # Path to the save initial parameters
            # otherwise, we generate a new set of params
            # with the net_gen
            if cfg.fed.load_saved_parameters:
                # Use the results_dir by default
                # otherwise use the specificed folder
                parameters_path = (
                    results_dir / "parameters"
                    if cfg.fed.use_results_dir
                    else Path(cfg.fed.parameters_folder)
                )
            else:
                # Generate new parameters
                parameters_path = None

            # Parameters for the strategy
            initial_parameters = get_initial_parameters(
                net_generator,
                cast(
                    dict, OmegaConf.to_container(cfg.task.net_config_initial_parameters)
                ),
                load_from=parameters_path,
                round=cfg.fed.parameters_round,
            )

            # Define your strategy
            # pass all relevant argument
            # Fraction_fit and fraction_evaluate are ignored
            # in favor of using absolute numbers via min_fit_clients
            # get_weighted_avg_metrics_agg_fn obeys
            # the fit_metrics and evaluate_metrics
            # in the cfg.task
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
                    cfg.task.fit_metrics
                ),
                evaluate_metrics_aggregation_fn=get_weighted_avg_metrics_agg_fn(
                    cfg.task.evaluate_metrics
                ),
                initial_parameters=initial_parameters,
            )

            # Server that handles Wandb and file saving
            server = WandbServer(
                client_manager=client_manager,
                history=history,
                strategy=strategy,
                save_parameters_to_file=save_parameters_to_file,
                save_files_per_round=save_files_per_round,
            )

            # Client generation function for Ray
            # Do not change
            client_generator: ClientGen = get_client_generator(
                working_dir=working_dir,
                net_generator=net_generator,
                dataloader_gen=client_dataloader_gen,
                train=train_func,
                test=test_func,
            )

            # Seed everything to maybe improve reproduceability
            seed_everything(adjusted_seed)

            # Runs fit and eval on either one client or all of them
            # Avoids launching ray for debugging purposes
            test_client(
                test_all_clients=cfg.test_clients.all,
                test_one_client=cfg.test_clients.one,
                client_generator=client_generator,
                initial_parameters=initial_parameters,
                total_clients=cfg.fed.num_total_clients,
                on_fit_config_fn=on_fit_config_fn,
                on_evaluate_config_fn=on_evaluate_config_fn,
            )

            # Start Simulation
            # The ray_init_args are only necessary
            # If multiple ray servers run in parallel
            # you should provide them from wherever
            # you start your server (e.g., sh script)
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

            # Make a dir for the histories
            histories_dir = working_dir / "histories"
            histories_dir.mkdir(parents=True, exist_ok=True)

            # Dump the json rather than the object
            with open(histories_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history.__dict__, f, ensure_ascii=False)

        # Sync the entire results dir to wandb if enabled
        # Only once at the end of the simulation
        if run is not None:
            run.save(
                str((results_dir / "*").resolve()),
                str((results_dir).resolve()),
                "now",
            )
            # Try to empty the wandb folder of old local runs
            log(
                logging.INFO,
                subprocess.run(
                    ["wandb", "sync", "--clean-old-hours", "24"],
                    capture_output=True,
                    text=True,
                    check=True,
                ),
            )


if __name__ == "__main__":
    main()
