"""Default training and testing functions, local and federated."""

from typing import Optional

from project.typing.common import FedEvalFN, OnEvaluateConfigFN, OnFitConfigFN


def get_fed_eval_fn() -> Optional[FedEvalFN]:
    """Define your evaluation function for the server here.

    Set all necessary parameters using the closure.
    """
    return None


def get_on_fit_config_fn() -> Optional[OnFitConfigFN]:
    """Define your on_fit_config_fn here.

    Set all necessary parameters using the closure.
    """
    return None


def get_on_evaluate_config_fn() -> Optional[OnEvaluateConfigFN]:
    """Define your on_evaluate_config_fn here.

    Set all necessary parameters using the closure.
    """
    return None
