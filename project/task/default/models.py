"""Define our models, and training and eval functions."""

import torch.nn as nn


class Net(nn.Module):
    """A PyTorch model."""

    # TODO: define your model here
    pass


def get_net(config: dict) -> nn.Module:
    """Return a model instance.

    Args:
        config: A dictionary with the model configuration.

    Returns
    -------
    nn.Module
        A PyTorch model.
    """
    return Net()
