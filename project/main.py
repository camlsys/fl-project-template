"""Create and connect the building blocks for your experiments; start the simulation.

It includes processing the dataset, instantiate strategy, specifying how the global
model will be evaluated, etc. In the end, this script saves the results.
"""

import copy
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import cast
import uuid

import flwr as fl
import hydra
import wandb
from wandb.sdk.wandb_run import Run
from flwr.common.logger import log
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from project.dispatch.dispatch import (
    dispatch_config,
    dispatch_data,
    dispatch_get_client_generator,
    dispatch_get_client_manager,
    dispatch_server,
    dispatch_train,
)
from project.fed.utils.utils import (
    get_save_history_to_file,
    get_state,
    get_save_parameters_to_file,
    get_save_rng_to_file,
    get_weighted_avg_metrics_agg_fn,
    test_client,
)
from project.types.common import ClientGen, FedEvalFN, Folders
from project.utils.utils import (
    FileSystemManager,
    RayContextManager,
    load_wandb_run_details,
    save_wandb_run_details,
    wandb_init,
)

# Make debugging easier when using Hydra + Ray
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["OC_CAUSE"] = "1"


@hydra.main(
    config_path="conf",
    config_name="base",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # Print parsed config
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    wandb_config = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )

    # Obtain the output dir from hydra
    original_hydra_dir = Path(
        hydra.utils.to_absolute_path(
            HydraConfig.get().runtime.output_dir,
        ),
    )

    output_directory = original_hydra_dir

    # The directory to save data to
    results_dir = output_directory / Folders.RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)

    # Where to save files to and from
    if cfg.working_dir is not None:
        # Pre-defined directory
        working_dir = Path(cfg.working_dir)
    else:
        # Default directory
        working_dir = output_directory / Folders.WORKING

    working_dir.mkdir(parents=True, exist_ok=True)

    # Restore wandb runs automatically
    wandb_id = None
    if cfg.use_wandb and cfg.wandb_resume:
        if cfg.wandb_id is not None:
            wandb_id = cfg.wandb_id
        elif (
            saved_wandb_details := load_wandb_run_details(results_dir / Folders.WANDB)
        ) is not None:
            wandb_id = saved_wandb_details.wandb_id

    # Wandb context manager
    # controls if wandb is initialized or not
    # if not it returns a dummy run
    with wandb_init(
        cfg.use_wandb,
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),
        config=wandb_config,
        resume="must" if cfg.wandb_resume and wandb_id is not None else "allow",
        id=wandb_id if wandb_id is not None else uuid.uuid4().hex,
    ) as run:
        if cfg.use_wandb:
            save_wandb_run_details(cast(Run, run), working_dir / Folders.WANDB)
        log(
            logging.INFO,
            "Wandb run initialized with %s",
            cfg.use_wandb,
        )

        # Context managers for saving and cleaning up files
        # from the working directory
        # at the start/end of the simulation
        # The RayContextManager deletes the ray session folder
        with (
            FileSystemManager(
                working_dir=working_dir,
                results_dir=results_dir,
                load_parameters_from=cfg.fed.parameters_folder,
                to_clean_once=cfg.to_clean_once,
                to_save_once=cfg.to_save_once,
                to_restore=cfg.to_restore,
                original_hydra_dir=original_hydra_dir,
                starting_round=cfg.fed.server_round,
                file_limit=int(cfg.file_limit),
            ) as fs_manager,
            RayContextManager() as _ray_manager,
        ):
            # Obtain the net generator, dataloader and fed_dataloader
            # Change the cfg.task.model_and_data str to change functionality
            (
                net_generator,
                initial_parameter_gen,
                client_dataloader_gen,
                fed_dataloader_gen,
                init_working_dir,
            ) = data_structure = dispatch_data(
                cfg,
            )
            # The folder starts either empty or only with restored files
            # as specified in the config
            if init_working_dir is not None:
                init_working_dir(working_dir, results_dir)

            # Parameters/rng/history state for the strategy
            # Uses the path to the saved initial parameters and state
            # If none are available, new ones will be generated

            # Use the results_dir by default
            # otherwise use the specified folder

            saved_state = get_state(
                net_generator,
                initial_parameter_gen,
                config=cast(
                    dict,
                    OmegaConf.to_container(
                        cfg.task.net_config_initial_parameters,
                    ),
                ),
                load_parameters_from=(
                    results_dir / Folders.STATE / Folders.PARAMETERS
                    if cfg.fed.parameters_folder is None
                    else Path(cfg.fed.parameters_folder)
                ),
                load_rng_from=(
                    results_dir / Folders.STATE / Folders.RNG
                    if cfg.fed.rng_folder is None
                    else Path(cfg.fed.rng_folder)
                ),
                load_history_from=(
                    results_dir / Folders.STATE / Folders.HISTORIES
                    if cfg.fed.history_folder is None
                    else Path(cfg.fed.history_folder)
                ),
                seed=cfg.fed.seed,
                server_round=fs_manager.server_round,
                use_wandb=cfg.use_wandb,
                hydra_config=cfg,
            )
            initial_parameters, server_rng, history = saved_state

            server_isolated_rng, client_cid_rng, client_seed_rng = server_rng

            # Client manager that samples the same clients
            # For a given seed+checkpoint combination
            client_manager = dispatch_get_client_manager(cfg)(
                enable_resampling=cfg.fed.enable_resampling,
                client_cid_generator=client_cid_rng,
                hydra_config=cfg,
            )

            # Obtain the train/test func and the fed eval func
            # Change the cfg.task.train_structure str to change functionality
            (
                train_func,
                test_func,
                get_fed_eval_fn,
            ) = train_structure = dispatch_train(cfg)

            # Obtain the on_fit config and on_eval config
            # generation functions
            # These depend on the cfg.task.fit_config
            # and cfg.task.eval_config dictionaries by default
            (
                on_fit_config_fn,
                on_evaluate_config_fn,
            ) = config_structure = dispatch_config(cfg)

            get_client_generator, actor_type, actor_kwargs = (
                dispatch_get_client_generator(
                    cfg,
                    saved_state=saved_state,
                    working_dir=working_dir,
                    data_structure=data_structure,
                    train_structure=train_structure,
                    config_structure=config_structure,
                )
            )

            # Build the evaluate function from the given components
            # This is the function that is called on the server
            # to evaluated the global model
            # the cast to Dict is necessary for mypy
            # as is the to_container
            evaluate_fn: FedEvalFN | None = get_fed_eval_fn(
                net_generator,
                fed_dataloader_gen,
                test_func,
                cast(
                    dict,
                    OmegaConf.to_container(
                        cfg.task.fed_test_config,
                    ),
                ),
                working_dir,
                server_isolated_rng,
                copy.deepcopy(cfg),
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
                    cfg.task.fit_metrics,
                ),
                evaluate_metrics_aggregation_fn=get_weighted_avg_metrics_agg_fn(
                    cfg.task.evaluate_metrics,
                ),
                initial_parameters=initial_parameters,
            )

            # Server that handles Wandb and file saving
            server = dispatch_server(cfg)(
                client_manager=client_manager,
                hydra_config=cfg,
                starting_round=fs_manager.server_round,
                server_rng=server_rng,
                history=history,
                strategy=strategy,
                save_parameters_to_file=get_save_parameters_to_file(
                    working_dir / Folders.STATE / Folders.PARAMETERS
                    if cfg.fed.parameters_folder is None
                    else Path(cfg.fed.parameters_folder)
                ),
                save_history_to_file=get_save_history_to_file(
                    working_dir / Folders.STATE / Folders.HISTORIES
                    if cfg.fed.history_folder is None
                    else Path(cfg.fed.history_folder)
                ),
                save_rng_to_file=get_save_rng_to_file(
                    working_dir / Folders.STATE / Folders.RNG
                    if cfg.fed.rng_folder is None
                    else Path(cfg.fed.rng_folder)
                ),
                save_files_per_round=fs_manager.get_save_files_every_round(
                    cfg.to_save_per_round,
                    cfg.save_frequency,
                ),
            )

            # Client generation function for Ray
            # Do not change
            client_generator: ClientGen = get_client_generator(
                working_dir,
                net_generator,
                client_dataloader_gen,
                train_func,
                test_func,
                client_seed_rng,
                cfg,
            )
            if initial_parameters is not None:
                # Runs fit and eval on either one client or all of them
                # Avoids launching ray for debugging purposes
                test_client(
                    test_all_clients=cfg.debug_clients.all,
                    test_one_client=cfg.debug_clients.one,
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
            # NOTE: `client_resources` accepts fractional
            # values for `num_cpus` and `num_gpus` iff
            # they're lower than 1.0.
            fl.simulation.start_simulation(
                # NOTE: mypy complains about the type of client_generator
                # We must wait for reconciliation from Flower
                client_fn=lambda cid: client_generator(cid).to_client(),
                num_clients=cfg.fed.num_total_clients,
                client_resources={
                    "num_cpus": (
                        int(
                            cfg.fed.cpus_per_client,
                        )
                        if cfg.fed.cpus_per_client >= 1
                        else float(
                            cfg.fed.cpus_per_client,
                        )
                    ),
                    "num_gpus": (
                        int(
                            cfg.fed.gpus_per_client,
                        )
                        if cfg.fed.gpus_per_client >= 1
                        else float(
                            cfg.fed.gpus_per_client,
                        )
                    ),
                },
                server=server,
                config=fl.server.ServerConfig(
                    num_rounds=cfg.fed.num_rounds,
                ),
                ray_init_args=(
                    {
                        "include_dashboard": False,
                        "address": cfg.ray_address,
                        "_redis_password": cfg.ray_redis_password,
                        "_node_ip_address": cfg.ray_node_ip_address,
                    }
                    if cfg.ray_address is not None
                    else {"include_dashboard": False}
                ),
                actor_type=actor_type,
                actor_kwargs=actor_kwargs,
            )

        # Sync the entire results dir to wandb if enabled
        # Only once at the end of the simulation
        if run is not None:
            run.save(
                str((results_dir / "*").resolve()),
                str((results_dir).resolve()),
                "now",
            )

            if cfg.fed.parameters_folder is not None:
                run.save(
                    str((Path(cfg.fed.parameters_folder) / "*").resolve()),
                    str((Path(cfg.fed.parameters_folder)).resolve()),
                    "now",
                )
            if cfg.fed.history_folder is not None:
                run.save(
                    str((Path(cfg.fed.history_folder) / "*").resolve()),
                    str((Path(cfg.fed.history_folder)).resolve()),
                    "now",
                )
            if cfg.fed.rng_folder is not None:
                run.save(
                    str((Path(cfg.fed.rng_folder) / "*").resolve()),
                    str((Path(cfg.fed.rng_folder)).resolve()),
                    "now",
                )

            # Try to empty the wandb folder of old local runs
            log(
                logging.INFO,
                subprocess.run(
                    [
                        "wandb",
                        "sync",
                        "--clean-old-hours",
                        "24",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ),
            )


if __name__ == "__main__":
    main()
