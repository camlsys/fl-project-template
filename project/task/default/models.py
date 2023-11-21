"""Define our models, and training and eval functions."""

from torch import nn


class Net(nn.Module):
    """A PyTorch model."""

    # TODO: define your model here


def get_net(_config: dict) -> nn.Module:
    """Return a model instance.

    Args:
        config: A dictionary with the model configuration.

    Returns
    -------
    nn.Module
        A PyTorch model.
    """
    return Net()
