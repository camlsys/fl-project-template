"""The default client implementation.

Make sure the model and dataset are not loaded before the fit function.
"""
from pathlib import Path
from typing import Dict

from flwr.common import NDArrays

from project.client.client import Client
from project.task.mnist_classification.train_test import test, train
from project.types.common import ClientDataloaderGen, ClientGen, EvalRes, FitRes, NetGen
from project.utils.utils import obtain_device


class ClientMNIST(Client):
    """MNIST virtual client for ray."""

    def __init__(
        self,
        cid: int | str,
        working_dir: Path,
        net_generator: NetGen,
        dataloader_gen: ClientDataloaderGen,
    ) -> None:
        """Initialize the client.

        Only ever instantiate the model or load dataset
        inside fit/eval, never in init.

        Parameters
        ----------
        cid : int | str
            The client's ID.
        working_dir : Path
            The path to the working directory.
        net_generator : NetGen
            The network generator.
        dataloader_gen : ClientDataloaderGen
            The dataloader generator.
            Uses the client id to determine partition.

        Returns
        -------
        None
        """
        super().__init__(cid, working_dir, net_generator, dataloader_gen)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> FitRes:
        """Fit the model using the provided parameters.

        Only ever instantiate the model or load dataset
        inside fit, never in init.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to use for training.
        config : Dict
            The configuration for the training.

        Returns
        -------
        FitRes
            The parameters after training, the number of samples used and the metrics.
        """
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
        """Evaluate the model using the provided parameters.

        Only ever instantiate the model or load dataset
        inside eval, never in init.

        Parameters
        ----------
        parameters : NDArrays
            The parameters to use for evaluation.
        config : Dict
            The configuration for the evaluation.

        Returns
        -------
        EvalRes
            The loss, the number of samples used and the metrics.
        """
        self.net = self.set_parameters(parameters, config)
        testloader = self.dataloader_gen(self.cid, True, config)
        loss, num_samples, metrics = test(self.net, testloader, obtain_device())
        return loss, num_samples, metrics


def get_client_generator(
    working_dir: Path, net_generator: NetGen, dataloader_gen: ClientDataloaderGen
) -> ClientGen:
    """Return a function which creates a new Client.

    Client has access to the working dir, can generate a network and
    load its own data.

    Parameters
    ----------
    working_dir : Path
        The path to the working directory.
    net_generator : NetGen
        The network generator.
    dataloader_gen : ClientDataloaderGen
        The dataloader generator.
        Uses the client id to determine partition.

    Returns
    -------
    Callable[[int | str], Client]
        The function to create a new Client.
    """
    return lambda i: ClientMNIST(
        i,
        working_dir=working_dir,
        net_generator=net_generator,
        dataloader_gen=dataloader_gen,
    )
