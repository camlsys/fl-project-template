"""Define our models, and training and eval functions."""

from omegaconf import DictConfig
from torch import nn

from project.types.common import IsolatedRNG


class Net(nn.Module):
    """A PyTorch model."""

    # TODO: define your model here


def get_net(
    _config: dict,
    rng_tuple: IsolatedRNG,
    _hydra_config: DictConfig | None,
) -> nn.Module:
    """Return a model instance.

    Args:
    config: A dictionary with the model configuration.
    rng_tuple: The random number generator state for the training.
        Use if you need seeded random behavior

    Returns
    -------
    nn.Module
        A PyTorch model.
    """
    return Net()
