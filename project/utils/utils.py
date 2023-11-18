"""Define any utility function.

Generic utilities.
"""

import logging
import random
import re
import shutil
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import ray
import torch
from flwr.common.logger import log

import wandb


def obtain_device() -> torch.device:
    """Get the device (CPU or GPU) for torch."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lazy_wrapper(x: Callable) -> Callable[[], Any]:
    """Wrap a value in a function that returns the value.

    For easy instantion through hydra.
    """
    return lambda: x


def lazy_config_wrapper(x: Callable) -> Callable[[Dict], Any]:
    """Wrap a value in a function that returns the value given a config.

    For easy instantion through hydra.
    """
    return lambda _config: x()


def seed_everything(seed: int) -> None:
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NoOpContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:
        """Do nothing."""
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        """Do nothing."""


def wandb_init(wandb_enabled: bool, *args, **kwargs):
    """Initialize wandb if enabled."""
    if wandb_enabled:
        return wandb.init(*args, **kwargs)

    return NoOpContextManager()


class RayContextManager:
    """A context manager for cleaning up after ray."""

    def __enter__(self):
        """Initialize the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Cleanup the files."""
        if ray.is_initialized():
            temp_dir = Path(
                ray.worker._global_node.get_session_dir_path()  # type: ignore
            )
            ray.shutdown()
            directory_size = shutil.disk_usage(temp_dir).used
            shutil.rmtree(temp_dir)
            log(
                logging.INFO,
                f"Cleaned up ray temp session: {temp_dir} with size: {directory_size}",
            )


def cleanup(working_dir: Path, to_clean: List[str]) -> None:
    """Cleanup the files in the working dir."""
    children: List[Path] = []
    for file in working_dir.iterdir():
        if file.is_file():
            for clean_token in to_clean:
                if clean_token in file.name:
                    if file.exists():
                        file.unlink()
                        break
        else:
            children.append(file)

    for child in children:
        cleanup(child, to_clean)


def get_checkpoint_index(output_dir: Path, file_limit: Optional[int]) -> int:
    """Get the index of the next checkpoint.

    Parameters
    ----------
    output_dir : Path
        The output directory.
        file_limit : Optional[int]
        The maximal number of files to search.
        If None, then there is no limit.

    Returns
    -------
        int
        The index of the next checkpoint.
    """
    same_name_files = chain(output_dir.glob("*_*"), output_dir.glob("*/*_*"))

    same_name_files = (
        same_name_files
        if file_limit is None
        else (next(same_name_files) for _ in range(file_limit))
    )

    indicies = (
        int(v.group(1))
        for f in same_name_files
        if (v := re.search(r"_([0-9]+)", f.stem))
    )

    return max(indicies, default=-1) + 1


def save_files(
    working_dir: Path,
    output_dir: Path,
    to_save: List[str],
    checkpoint_index: int,
    ending: Optional[int] = None,
    top_level: bool = True,
) -> None:
    """Save the files in the working dir."""
    if not top_level:
        output_dir = output_dir / working_dir.name

    children: List[Path] = []
    for file in working_dir.iterdir():
        if file.is_file():
            for save_token in to_save:
                if save_token in file.name:
                    if file.exists():
                        true_ending = (
                            f"{checkpoint_index}" + ("_" + str(ending))
                            if ending is not None
                            else f"{checkpoint_index}"
                        )
                        destination_file = (
                            output_dir
                            / file.with_stem(f"{file.stem}_{true_ending}").name
                        )

                        destination_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(file, destination_file)
                        break
        else:
            children.append(file)

    for child in children:
        save_files(
            child,
            output_dir,
            to_save=to_save,
            ending=ending,
            top_level=False,
            checkpoint_index=checkpoint_index,
        )


class FileSystemManager:
    """A context manager for saving and cleaning up files."""

    def __init__(
        self,
        working_dir: Path,
        output_dir,
        to_clean_once: List[str],
        to_save_once: List[str],
        original_hydra_dir: Path,
        reuse_output_dir: bool,
        file_limit: Optional[int] = None,
    ) -> None:
        self.to_clean_once = to_clean_once
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.to_save_once = to_save_once
        self.original_hydra_dir = original_hydra_dir
        self.reuse_output_dir = reuse_output_dir
        self.checkpoint_index = get_checkpoint_index(self.output_dir, file_limit)

    def get_save_files_every_round(
        self,
        to_save: List[str],
        save_frequency: int,
    ) -> Callable[[int], None]:
        """Get a function that saves files every save_frequency rounds."""

        def save_files_round(cur_round: int) -> None:
            if cur_round % save_frequency == 0:
                save_files(
                    self.working_dir,
                    self.output_dir,
                    to_save=to_save,
                    ending=cur_round,
                    checkpoint_index=self.checkpoint_index,
                )

        return save_files_round

    def __enter__(self):
        """Initialize the context manager and cleanup."""
        log(logging.INFO, f"Pre-cleaning {self.to_clean_once}")
        cleanup(self.working_dir, self.to_clean_once)

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Cleanup the files."""
        log(logging.INFO, f"Saving {self.to_save_once}")

        # Copy the hydra directory to the working directory
        # so that multiple runs can be ran
        # in the same output directory and configs versioned
        hydra_dir = self.working_dir / ".hydra"

        shutil.copytree(
            str(self.original_hydra_dir / ".hydra"),
            str(object=hydra_dir),
            dirs_exist_ok=True,
        )

        # Move main.log to the working directory
        main_log = self.original_hydra_dir / "main.log"
        shutil.copy2(str(main_log), str(self.working_dir / "main.log"))
        save_files(
            self.working_dir,
            self.output_dir,
            to_save=self.to_save_once,
            checkpoint_index=self.checkpoint_index,
        )
        log(logging.INFO, f"Post-cleaning {self.to_clean_once}")
        cleanup(self.working_dir, to_clean=self.to_clean_once)


def get_parameter_convertor(
    convertors: Iterable[Tuple[Any, Callable]]
) -> Callable[[Callable], Callable]:
    """Get a decorator that converts parameters to the right type."""

    def convert(param: Any) -> bool:
        for param_type, convertor in convertors:
            if isinstance(param, param_type):
                return convertor(param)
        return param

    def convert_params(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            new_args = []
            new_kwargs = {}
            for arg in args:
                new_args.append(convert(arg))
            for kwarg_name, kwarg_value in kwargs.items():
                new_kwargs[kwarg_name] = convert(kwarg_value)
            return func(*new_args, **new_kwargs)

        return wrapper

    return convert_params
