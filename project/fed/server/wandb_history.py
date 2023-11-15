"""History class which sends metrics to wandb.

This means metrics are collected only at the central server, minimizing communication
costs. Metric collection only happens if wandb is turned on.
"""
from typing import Dict

from flwr.common.typing import Scalar
from flwr.server.history import History

import wandb


class WandbHistory(History):
    """History class for training and/or evaluation metrics collection."""

    def __init__(self, use_wandb: bool = True) -> None:
        super().__init__()
        self.use_wandb = use_wandb

    def add_loss_distributed(self, server_round: int, loss: float) -> None:
        """Add one loss entry (from distributed evaluation)."""
        super().add_loss_distributed(server_round, loss)
        if self.use_wandb:
            wandb.log({"distributed_loss": loss}, step=server_round)

    def add_loss_centralized(self, server_round: int, loss: float) -> None:
        """Add one loss entry (from centralized evaluation)."""
        super().add_loss_centralized(server_round, loss)
        if self.use_wandb:
            wandb.log({"centralised_loss": loss}, step=server_round)

    def add_metrics_distributed_fit(
        self, server_round: int, metrics: Dict[str, Scalar]
    ) -> None:
        """Add metrics entries (from distributed fit)."""
        super().add_metrics_distributed_fit(server_round, metrics)
        if self.use_wandb:
            for key in metrics:
                wandb.log({key: metrics[key]}, step=server_round)

    def add_metrics_distributed(
        self, server_round: int, metrics: Dict[str, Scalar]
    ) -> None:
        """Add metrics entries (from distributed evaluation)."""
        super().add_metrics_distributed(server_round, metrics)
        if self.use_wandb:
            for key in metrics:
                wandb.log({key: metrics[key]}, step=server_round)

    def add_metrics_centralized(
        self, server_round: int, metrics: Dict[str, Scalar]
    ) -> None:
        """Add metrics entries (from centralized evaluation)."""
        super().add_metrics_centralized(server_round, metrics)
        if self.use_wandb:
            for key in metrics:
                wandb.log({key: metrics[key]}, step=server_round)
