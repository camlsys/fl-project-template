"""Flower server accounting for Weights&Biases+file saving."""

import timeit
from collections.abc import Callable
from logging import INFO

from flwr.common import Parameters
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy

from project.types.common import ServerRNG


class WandbServer(Server):
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        starting_round: int = 0,
        server_rng: ServerRNG,
        strategy: Strategy | None = None,
        history: History | None = None,
        save_parameters_to_file: Callable[
            [Parameters],
            None,
        ],
        save_rng_to_file: Callable[[ServerRNG], None],
        save_history_to_file: Callable[[History], None],
        save_files_per_round: Callable[[int], None],
    ) -> None:
        """Flower server implementation.

        Parameters
        ----------
        client_manager : ClientManager
            Client manager implementation.
        strategy : Optional[Strategy]
            Strategy implementation.
        history : Optional[History]
            History implementation.
        save_parameters_to_file : Callable[[Parameters], None]
            Function to save the parameters to file.
        save_files_per_round : Callable[[int], None]
            Function to save files every round.

        Returns
        -------
        None
        """
        super().__init__(
            client_manager=client_manager,
            strategy=strategy,
        )

        self.history: History | None = history
        self.save_parameters_to_file = save_parameters_to_file
        self.save_files_per_round = save_files_per_round
        self.starting_round = starting_round
        self.server_rng = server_rng
        self.save_rng_to_file = save_rng_to_file
        self.save_history_to_file = save_history_to_file

    # pylint: disable=too-many-locals
    def fit(
        self,
        num_rounds: int,
        timeout: float | None,
    ) -> History:
        """Run federated averaging for a number of rounds.

        Parameters
        ----------
        num_rounds : int
            The number of rounds to run.
        timeout : Optional[float]
            Timeout in seconds.

        Returns
        -------
        History
            The history of the training.
            Potentially using a pre-defined history.
        """
        history = self.history if self.history is not None else History()

        if self.starting_round == 0:
            # Initialize parameters
            log(INFO, "Initializing global parameters")
            self.parameters = self._get_initial_parameters(
                timeout=timeout,
            )
            log(INFO, "Evaluating initial parameters")
            res = self.strategy.evaluate(
                0,
                parameters=self.parameters,
            )
            if res is not None:
                log(
                    INFO,
                    "initial parameters (loss, other metrics): %s, %s",
                    res[0],
                    res[1],
                )
                history.add_loss_centralized(
                    server_round=0,
                    loss=res[0],
                )
                history.add_metrics_centralized(
                    server_round=0,
                    metrics=res[1],
                )
            # Save initial parameters and files
            self.save_parameters_to_file(self.parameters)
            self.save_rng_to_file(self.server_rng)
            self.save_files_per_round(0)

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(self.starting_round + 1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                (
                    parameters_prime,
                    fit_metrics,
                    _,
                ) = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round,
                    metrics=fit_metrics,
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(
                current_round,
                parameters=self.parameters,
            )
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(
                    server_round=current_round,
                    loss=loss_cen,
                )
                history.add_metrics_centralized(
                    server_round=current_round,
                    metrics=metrics_cen,
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round,
                        loss=loss_fed,
                    )
                    history.add_metrics_distributed(
                        server_round=current_round,
                        metrics=evaluate_metrics_fed,
                    )
            # Saver round parameters and files
            self.save_parameters_to_file(self.parameters)
            self.save_history_to_file(history)
            self.save_rng_to_file(self.server_rng)
            self.save_files_per_round(current_round)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
