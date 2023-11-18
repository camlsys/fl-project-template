"""The default client implementation.

Make sure the model and dataset are not loaded before the fit function.
"""
from pathlib import Path
from typing import Callable, Dict

from flwr.common import NDArrays

from project.client.client import Client
from project.task.mnist_classification.train_test import test, train
from project.types.common import ClientDataloaderGen, EvalRes, FitRes, NetGen
from project.utils.utils import obtain_device


class ClientMNIST(Client):
    """Virtual client for ray."""

    def __init__(
        self,
        cid: int | str,
        working_dir: Path,
        net_generator: NetGen,
        dataloader_gen: ClientDataloaderGen,
    ) -> None:
        super().__init__(cid, working_dir, net_generator, dataloader_gen)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> FitRes:
        """Fit the model using the provided parameters."""
        self.net = self.set_parameters(parameters, config)
        trainloader = self.dataloader_gen(self.cid, False, config)
        num_samples, metrics = train(
            self.net,
            trainloader,
            obtain_device(),
            config["epochs"],
            config["learning_rate"],
        )
        return self.get_parameters(config), num_samples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> EvalRes:
        """Evaluate the model using the provided parameters."""
        self.net = self.set_parameters(parameters, config)
        testloader = self.dataloader_gen(self.cid, True, config)
        loss, num_samples, metrics = test(self.net, testloader, obtain_device())
        return loss, num_samples, metrics


def get_client_generator(
    working_dir: Path, net_generator: NetGen, dataloader_gen: ClientDataloaderGen
) -> Callable[[int | str], Client]:
    """Return a function which creates a new Client for a given config."""
    return lambda i: ClientMNIST(
        i,
        working_dir=working_dir,
        net_generator=net_generator,
        dataloader_gen=dataloader_gen,
    )
