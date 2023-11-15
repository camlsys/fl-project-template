"""Define any utility function.

Generic utilities.
"""


import logging
import random
import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import ray
import torch
from flwr.common.logger import log

import wandb


def get_device() -> torch.device:
    """Get the device (CPU or GPU) for torch."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lazy_wrapper(x: Callable) -> Callable[[], Any]:
    """Wrap a value in a function that returns the value.

    For easy instantion through hydra.
    """
    return lambda: x


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


def save_files(
    working_dir: Path,
    output_dir: Path,
    to_save: List[str],
    ending: Optional[str] = None,
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
                        if ending is not None:
                            destination_file = (
                                output_dir / file.with_stem(f"{file.stem}{ending}").name
                            )
                        else:
                            destination_file = output_dir / file.name

                        destination_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(file, destination_file)
                        break
        else:
            children.append(file)

    for child in children:
        save_files(child, output_dir, to_save=to_save, ending=ending, top_level=False)


def get_save_files_every_round(
    working_dir: Path,
    output_dir: Path,
    to_save: List[str],
    save_frequency: int,
) -> Callable[[int], None]:
    """Get a function that saves files every save_frequency rounds."""

    def save_files_round(cur_round: int) -> None:
        if cur_round % save_frequency == 0:
            save_files(working_dir, output_dir, to_save=to_save, ending=f"_{cur_round}")

    return save_files_round


class FileSystemManager:
    """A context manager for saving and cleaning up files."""

    def __init__(
        self,
        working_dir: Path,
        output_dir,
        to_clean_once: List[str],
        to_save_once: List[str],
    ) -> None:
        self.to_clean_once = to_clean_once
        self.path_dict = working_dir
        self.output_dir = output_dir
        self.to_save_once = to_save_once

    def __enter__(self):
        """Initialize the context manager and cleanup."""
        log(logging.INFO, f"Pre-cleaning {self.to_clean_once}")
        cleanup(self.path_dict, self.to_clean_once)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Cleanup the files."""
        log(logging.INFO, f"Saving {self.to_save_once}")
        save_files(
            self.path_dict, self.output_dir, to_save=self.to_save_once, ending=""
        )
        log(logging.INFO, f"Post-cleaning {self.to_clean_once}")
        cleanup(self.path_dict, to_clean=self.to_clean_once)
