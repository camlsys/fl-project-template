"""Dynamically dispatch functionality from configs.

Handles mappign from strings specified in the hydra config to the task functions used in
the project. Dispatching at this top level should simply select the correct functions
from each task.
"""
