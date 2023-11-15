"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""
from typing import Dict

import torch.nn as nn


class Model(nn.Module):
    """A PyTorch model."""

    # TODO: define your model here
    pass


def get_model(config: Dict) -> nn.Module:
    """Return a model instance.

    Args:
        config: A dictionary with the model configuration.

    Returns
    -------
        A PyTorch model.
    """
    return Model()


def get_initial_parameters(config: Dict) -> nn.Module:
    """Return a model instance.

    Args:
        config: A dictionary with the model configuration.

    Returns
    -------
        A PyTorch model.
    """
    return Model()
