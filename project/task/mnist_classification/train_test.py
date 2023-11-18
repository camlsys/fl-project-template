"""MNIST training and testing functions, local and federated."""

from typing import Dict, Optional, Sized, Tuple, cast

import torch
from flwr.common import NDArrays
from torch import nn
from torch.utils.data import DataLoader

from project.fed.utils.utils import generic_set_parameters
from project.types.common import FedEvalFN, NetGen, OnFitConfigFN
from project.utils.utils import obtain_device


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> Tuple[int, Dict]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.

    Returns
    -------
    Tuple[int, Dict]
        The number of samples used for training,
        the loss, and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, trainloader.dataset)) == 0:
        raise ValueError("Trainloader can't be 0, exiting...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    net.train()
    final_epoch_per_sample_loss = 0.0
    num_correct = 0
    for _ in range(epochs):
        final_epoch_per_sample_loss = 0.0
        num_correct = 0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            final_epoch_per_sample_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()
            loss.backward()
            optimizer.step()

    return len(cast(Sized, trainloader.dataset)), {
        "train_loss": final_epoch_per_sample_loss
        / len(cast(Sized, trainloader.dataset)),
        "train_accuracy": float(num_correct) / len(cast(Sized, trainloader.dataset)),
    }


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, int, Dict]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, int, float]
        The loss, number of test samples,
        and the accuracy of the input model on the given data.
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        raise ValueError("Testloader can't be 0, exiting...")

    criterion = nn.CrossEntropyLoss()
    correct, per_sample_loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            per_sample_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    per_sample_loss /= len(cast(Sized, testloader.dataset))
    return (
        per_sample_loss / len(cast(Sized, testloader.dataset)),
        len(cast(Sized, testloader.dataset)),
        {"test_accuracy": float(correct) / len(cast(Sized, testloader.dataset))},
    )


def get_fed_eval_fn(
    net_generator: NetGen, testloader: DataLoader
) -> Optional[FedEvalFN]:
    """Get the federated evaluation function for MNIST.

    Parameters
    ----------
    net_generator : NetGenerator
        The function to generate the network.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.

    Returns
    -------
    Optional[FedEvalFN]
        The evaluation function for the server
        if the testloader is not empty, else None.
    """
    if len(cast(Sized, testloader.dataset)) == 0:
        return None

    def fed_eval_fn(
        _server_round: int, parameters: NDArrays, config: Dict
    ) -> Optional[Tuple[float, Dict]]:
        """Evaluate the model on the given data.

        Parameters
        ----------
        server_round : int
            The current server round.
        parameters : NDArrays
            The parameters of the model to evaluate.
        _config : Dict
            The configuration for the evaluation.

        Returns
        -------
        Optional[Tuple[float, Dict]]
            The loss and the accuracy of the input model on the given data.
        """
        net = net_generator(config)
        generic_set_parameters(net, parameters)
        net.eval()
        device = obtain_device()
        net.to(device)
        loss, num_samples, metrics = test(net, testloader, device)
        return loss, metrics

    return fed_eval_fn


def get_on_fit_config_fn(fit_config: Dict) -> Optional[OnFitConfigFN]:
    """MNIST on_fit_config_fn generator.

    Parameters
    ----------
    fit_config : Dict
        The configuration for the fit function.
        Loaded dynamically from the config file.

    Returns
    -------
    Optional[OnFitConfigFN]
        The on_fit_config_fn for the server if the fit_config is not empty, else None.
    """

    def fit_config_fn(server_round: int) -> Dict:
        """MNIST on_fit_config_fn.

        Parameters
        ----------
        server_round : int
            The current server round.
            Passed to the client

        Returns
        -------
        Dict
            The configuration for the fit function.
            Loaded dynamically from the config file.
        """
        # resolve and convert to python dict
        fit_config["curr_round"] = server_round  # add round info
        return fit_config

    return fit_config_fn


# Differences between the two will come
# from the config file
get_on_evaluate_config_fn = get_on_fit_config_fn
