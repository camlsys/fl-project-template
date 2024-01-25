"""Define any utility function.

Generic utilities.
"""

import json
import logging
import re
import shutil
from collections.abc import Callable, Iterator
from itertools import chain, islice
from pathlib import Path
from types import TracebackType
from typing import Any, cast
from pydantic import BaseModel

import ray
import torch
from flwr.common.logger import log
from project.fed.utils.utils import Files
import wandb
from wandb.sdk.wandb_run import Run
from wandb.sdk.lib.disabled import RunDisabled

from project.types.common import Ext, FileCountExceededError, Folders, IsolatedRNG


def obtain_device() -> torch.device:
    """Get the device (CPU or GPU) for torch.

    Returns
    -------
    torch.device
        The device.
    """
    return torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu",
    )


def lazy_wrapper(x: Callable) -> Callable[[], Any]:
    """Wrap a value in a function that returns the value.

    For easy instantion through hydra.

    Parameters
    ----------
    x : Callable
        The value to wrap.

    Returns
    -------
    Callable[[], Any]
        The wrapped value.
    """
    return lambda: x


def lazy_config_wrapper(x: Callable) -> Callable[[dict, IsolatedRNG], Any]:
    """Wrap a value in a function that returns the value given a config and rng_tuple.

    For easy instantiation through hydra.

    Parameters
    ----------
    x : Callable
        The value to wrap.

    Returns
    -------
    Callable[[Dict], Any]
        The wrapped value.
    """
    return lambda _config, _rng_tuple: x()


class NoOpContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:
        """Do nothing."""
        return

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Do nothing."""


def wandb_init(
    wandb_enabled: bool,
    *args: Any,
    **kwargs: Any,
) -> NoOpContextManager | Run | RunDisabled:
    """Initialize wandb if enabled.

    Parameters
    ----------
    wandb_enabled : bool
        Whether wandb is enabled.
    *args : Any
        The arguments to pass to wandb.init.
    **kwargs : Any
        The keyword arguments to pass to wandb.init.

    Returns
    -------
    Optional[Union[NoOpContextManager, Any]]
        The wandb context manager if enabled, otherwise a no-op context manager
    """
    if wandb_enabled:
        run = wandb.init(*args, **kwargs)
        if run is not None:
            return run

    return NoOpContextManager()


class WandbDetails(BaseModel):
    """The wandb details."""

    wandb_id: str


def save_wandb_run_details(run: Run, wandb_dir: Path) -> None:
    """Save the wandb run to the output directory.

    Parameters
    ----------
    run : Run
        The wandb run.
    wandb_dir : Path
        The output directory.

    Returns
    -------
        None
    """
    wandb_run_details: dict[str, str] = {
        "wandb_id": run.id,
    }

    # Check if it conforms to the WandbDetails schema
    WandbDetails(**wandb_run_details)

    wandb_dir.mkdir(parents=True, exist_ok=True)
    with open(
        wandb_dir / f"{Files.WANDB_RUN}.{Ext.WANDB_RUN}",
        mode="w",
        encoding="utf-8",
    ) as f:
        json.dump(wandb_run_details, f)


def load_wandb_run_details(wandb_dir: Path) -> WandbDetails | None:
    """Save the wandb run to the wandb_dir directory.

    Parameters
    ----------
    run : Run
        The wandb run.
    wandb_dir : Path
        The output directory.

    Returns
    -------
        None
    """
    wandb_file = wandb_dir / f"{Files.WANDB_RUN}.{Ext.WANDB_RUN}"

    if not wandb_file.exists():
        return None

    with open(
        wandb_file,
        encoding="utf-8",
    ) as f:
        return WandbDetails(**json.load(f))


class RayContextManager:
    """A context manager for cleaning up after ray."""

    def __enter__(self) -> "RayContextManager":
        """Initialize the context manager."""
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Cleanup the files.

        Parameters
        ----------
        _exc_type : Any
            The exception type.
        _exc_value : Any
            The exception value.
        _traceback : Any
            The traceback.

        Returns
        -------
        None
        """
        if ray.is_initialized():
            temp_dir = Path(
                ray.worker._global_node.get_session_dir_path(),
            )
            ray.shutdown()
            directory_size = shutil.disk_usage(
                temp_dir,
            ).used
            shutil.rmtree(temp_dir)
            log(
                logging.INFO,
                f"Cleaned up ray temp session: {temp_dir} with size: {directory_size}",
            )


def cleanup(working_dir: Path, to_clean: list[str]) -> None:
    """Cleanup the files in the working dir.

    Parameters
    ----------
    working_dir : Path
        The working directory.
    to_clean : List[str]
        The tokens to clean.

    Returns
    -------
        None
    """
    children: list[Path] = []
    for file in working_dir.iterdir():
        if file.is_file():
            for clean_token in to_clean:
                if clean_token in file.name and file.exists():
                    file.unlink()
                    break
        else:
            children.append(file)

    for child in children:
        cleanup(child, to_clean)


def get_highest_round(
    parameters_dir: Path,
    file_limit: int,
) -> int:
    """Get the index of the highest round.

    Parameters
    ----------
    output_dir : Path
        The output directory.
    file_limit : int
        The maximal number of files to search.
        If None, then there is no limit.

    Returns
    -------
    int
        The index of the highest round.
    """
    same_name_files = cast(
        Iterator[Path],
        islice(
            chain(
                parameters_dir.glob(f"*{Files.PARAMETERS}_*"),
                parameters_dir.glob(f"*/*{Files.PARAMETERS}_*"),
            ),
            file_limit,
        ),
    )

    indicies = (
        int(v.group(1))
        for f in same_name_files
        if (v := re.search(r"_([0-9]+)", f.stem))
    )
    return max(indicies, default=0)


def save_files(
    working_dir: Path,
    output_dir: Path,
    to_save: list[str],
    server_round: int,
    file_limit: int,
    top_level: bool = True,
    file_cnt: int = 0,
) -> None:
    """Save the files in the working dir.

    Parameters
    ----------
    working_dir : Path
        The working directory.
    output_dir : Path
        The output directory.

    Returns
    -------
        None
    """
    if not top_level:
        output_dir = output_dir / working_dir.name

    children: list[Path] = []
    for file in working_dir.iterdir():
        if file.is_file():
            for save_token in to_save:
                if save_token in file.name and file.exists():
                    # Save the round file
                    destination_file = (
                        output_dir
                        / file.with_stem(
                            f"{file.stem}_{server_round}",
                        ).name
                    )

                    latest_file = (
                        output_dir
                        / file.with_stem(
                            f"{file.stem}",
                        ).name
                    )

                    destination_file.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    shutil.copy(file, destination_file)
                    shutil.copy(file, latest_file)
                    break
        else:
            children.append(file)

    for child in children:
        save_files(
            child,
            output_dir,
            to_save=to_save,
            top_level=False,
            server_round=server_round,
            file_limit=file_limit,
            file_cnt=file_cnt,
        )


def restore_files(
    working_dir: Path,
    output_dir: Path,
    to_restore: list[str],
    server_round: int,
    file_limit: int,
    top_level: bool = True,
    file_cnt: int = 0,
) -> None:
    """Save the files in the working dir.

    Parameters
    ----------
    working_dir : Path
        The working directory.
    output_dir : Path
        The output directory.

    Returns
    -------
        None
    """
    if not top_level:
        working_dir = working_dir / output_dir.name

    children: list[Path] = []
    for file in output_dir.iterdir():
        file_cnt += 1
        if file.is_file():
            if f"_{server_round}" in file.name:
                for restore_token in to_restore:
                    if restore_token in file.name:
                        destination_file = (
                            working_dir
                            / file.with_stem(
                                f"{file.stem.replace(f'_{server_round}', '')}",
                            ).name
                        )

                        destination_file.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        shutil.copy(file, destination_file)
                        break
        else:
            children.append(file)
        if file_cnt >= file_limit:
            raise FileCountExceededError(
                f"""You have exceeded the {file_limit} file limit,
                you may increase it in the config if you are sure about it."""
            )

    for child in children:
        restore_files(
            working_dir,
            child,
            to_restore=to_restore,
            top_level=False,
            server_round=server_round,
            file_limit=file_limit,
            file_cnt=file_cnt,
        )


class FileSystemManager:
    """A context manager for saving and cleaning up files."""

    def __init__(
        self,
        working_dir: Path,
        results_dir: Path,
        load_parameters_from: Path | None,
        to_restore: list[str],
        to_clean_once: list[str],
        to_save_once: list[str],
        original_hydra_dir: Path,
        file_limit: int,
        starting_round: int | None,
    ) -> None:
        """Initialize the context manager.

        Parameters
        ----------
        working_dir : Path
            The working directory.
        results_dir : Path
            The output directory.
        to_clean_once : List[str]
            The tokens to clean once.
        to_save_once : List[str]
            The tokens to save once.
        original_hydra_dir : Path
            The original hydra directory.
            For copying the hydra directory to the working directory.
        file_limit : Optional[int]
            The maximal number of files to search.
            If None, then there is no limit.

        Returns
        -------
            None
        """
        self.to_clean_once = to_clean_once
        self.working_dir = working_dir
        self.results_dir = results_dir
        self.to_save_once = to_save_once
        self.to_restore = to_restore

        self.original_hydra_dir = original_hydra_dir

        highest_round = get_highest_round(
            parameters_dir=(
                load_parameters_from
                if load_parameters_from is not None
                else results_dir / Folders.STATE / Folders.PARAMETERS
            ),
            file_limit=file_limit,
        )
        self.file_limit = file_limit

        self.server_round = (
            min(
                highest_round,
                starting_round,
            )
            if starting_round is not None
            else highest_round
        )

    def get_save_files_every_round(
        self,
        to_save: list[str],
        save_frequency: int,
    ) -> Callable[[int], None]:
        """Get a function that saves files every save_frequency rounds.

        Parameters
        ----------
        to_save : List[str]
            The tokens to save.
        save_frequency : int
            The frequency to save.

        Returns
        -------
        Callable[[int], None]
            The function that saves the files.
        """

        def save_files_round(cur_round: int) -> None:
            self.server_round = cur_round
            if cur_round % save_frequency == 0:
                save_files(
                    self.working_dir,
                    self.results_dir,
                    to_save=to_save,
                    server_round=cur_round,
                    file_limit=self.file_limit,
                )

        return save_files_round

    def __enter__(self) -> "FileSystemManager":
        """Initialize the context manager and cleanup."""
        log(
            logging.INFO,
            f"Pre-cleaning {self.to_clean_once}",
        )
        cleanup(self.working_dir, self.to_clean_once)
        restore_files(
            self.working_dir,
            self.results_dir,
            self.to_restore,
            server_round=self.server_round,
            file_limit=self.file_limit,
        )
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Cleanup the files."""
        log(logging.INFO, f"Saving {self.to_save_once}")

        # Copy the hydra directory to the working directory
        # so that multiple runs can be ran
        # in the same output directory and configs versioned
        hydra_dir = self.working_dir / Folders.HYDRA

        shutil.copytree(
            str(self.original_hydra_dir / Folders.HYDRA),
            str(object=hydra_dir),
            dirs_exist_ok=True,
        )

        # Move main.log to the working directory
        main_log = self.original_hydra_dir / f"{Files.MAIN}.{Ext.MAIN}"
        shutil.copy2(
            str(main_log),
            str(self.working_dir / f"{Files.MAIN}.{Ext.MAIN}"),
        )
        save_files(
            self.working_dir,
            self.results_dir,
            to_save=self.to_save_once,
            server_round=self.server_round,
            file_limit=self.file_limit,
        )
        log(
            logging.INFO,
            f"Post-cleaning {self.to_clean_once}",
        )
        cleanup(
            self.working_dir,
            to_clean=self.to_clean_once,
        )
