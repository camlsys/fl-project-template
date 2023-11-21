
# [CaMLSys](https://mlsys.cst.cam.ac.uk/) Federated Learning Research Template using [Flower](https://github.com/adap/flower), [Hydra](https://github.com/facebookresearch/hydra), and [Wandb](https://wandb.ai/site)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/camlsys/fl-project-template/main.svg)](https://results.pre-commit.ci/latest/github/camlsys/fl-project-template/main)

> Note: If you use this template in your work, please reference it and cite the [Flower](https://arxiv.org/abs/2007.14390) paper.

> :warning: This ``README`` describes how to use the template as-is after installing it with the default `setup.sh` script in a machine running Ubuntu `22.04`. Please follow the instructions in `EXTENDED_README.md` for details on more complex environment setups and how to extend this template.

## About this Template

Federated Learning (FL) is a privacy-preserving machine learning paradigm that allows training models directly on local client data using local client resources. This template standardizes the FL research workflow at the [Cambridge ML Systems](https://mlsys.cst.cam.ac.uk/) based on three frameworks chosen for their flexibility and ease of use:
 - [Flower](https://github.com/adap/flower): The FL framework developed by [Flower Labs](https://flower.dev/) with contributions from [CaMLSys](https://mlsys.cst.cam.ac.uk/) members. 
 - [Hydra](https://github.com/facebookresearch/hydra): framework for managing experiments developed at Meta which automatically handles experimental configuration for Python.
 - [Wandb](https://wandb.ai/site): The MLOps platform developed for handling results storage, experiment tracking, reproducibility, and visualization.

While these tools can be combined in an ad-hoc manner, this template intends to provide a unified and opinionated structure for achieving this while providing functionality that may not have been easily constructed from scratch. 

### What this template does:
 - Automatically handles client configuration for Flower in an opinionated manner using the [PyTorch](https://github.com/pytorch/pytorch) library. This is meant to reduce the task of FL simulation to the mere implementation of standard ML tasks combined with minimal configuration work. Specifically, clients are treated uniformly except for their data, model, and configuration.
    - A user only needs to provide:
        - A means of generating a model (e.g., a function which returns a PyTorch model) based on a received configuration (e.g., a Dict)
        - A means of constructing train and test dataloaders
        - A means of offering a configuration to these components
    - All data loading or model training is delayed as much as possible to facilitate creating many clients and keeping them in memory with the smallest footprint possible.
    - Metric collection and aggregation require no additional implementation. 
- Automatically handles logging, saving, and checkpointing, which integrate natively and seamlessly with Wandb and Hydra. This enables sequential re-launches of the same job on clusters using time-limited schedulers.
- Provides deterministic seeded client selection while taking into account the current checkpoint. 
- Provides a static means of selecting which ML task to run using Hydra's config system without the drawbacks of the untyped mechanism provided by Hydra.
- Enforces good coding standards by default using isort, black, docformatter, ruff and mypy integrated with [pre-commit](https://pre-commit.com/). [Pydantic](https://docs.pydantic.dev/latest/) is also used to validate configuration data for generating models, creating dataloaders, training clients, etc.

### What this template does not do:
- Provide off-the-shelf implementations of FL algorithms, ML tasks, datasets, or models beyond the MNIST example. For such functionality, please refer to the original [Flower](https://github.com/adap/flower) and [PyTorch](https://github.com/pytorch/pytorch).
- Provide a means of running experiments on clusters as this depends on the cluster configuration.

## Setup

The basic setup has been simplified to one setup.sh script using [poetry](https://python-poetry.org/), [pyenv](https://github.com/pyenv/pyenv) and [pre-commit](https://pre-commit.com/). It only requires limited user input regarding the installation location of ``pyenv`` and ``poetry``, and will install the specified python version. All dependencies are placed in the local ``.venv`` directory. 


By default, pre-commit only runs hooks on files staged for commit. If you wish to run all the pre-commit hooks without committing or pushing, use:
```bash  
poetry run pre-commit run --all-files --hook-stage push
```


## Using the Template



> Note: these instructions rely on the MNIST task and assume specific dataset partitioning, model creation and dataloader instantiation procedure. We recommend following a similar structure in your own experiments. Please refer to the [Flower](https://flower.dev/docs/baselines/index.html) baselines for more examples. 

Install the template using the setup.sh script:
```bash
./setup.sh 
```




If ``poetry``, ``pyenv``, and/or the correct python version are installed, they will not be installed again. If not installed, you must provide paths to the desired install locations. If running on a cluster, this would be the location of the shared file system.

> :warning: Run the `default` task to check that everything is installed correctly from the root ``fl-project-template``, not from the ``fl-project-template/project`` directory.

```bash
poetry run python -m project.main --config-name=base
```


If you have a cluster which may run multiple Ray simulator instances, you will need to launch the server separately. 

The default task should have created a folder in fl-project-template/outputs. This folder contains the results of the experiment. To log your experiments to wandb, log into wandb and then enable it via the command:

```bash
poetry run python -m project.main --config-name=base use_wandb=true
```

Now you  can run the MNIST example by following these instructions:
- Specify a ``dataset_dir`` and ``partition_dir`` in ``conf/dataset/mnist.yaml`` together with the ``num_clients``, the size of a clients validation set ``val_ratio``, a ``seed`` for partitioning. Additionally, you can also specify if the partition labels should be ``iid``, follow a ``power_law`` distribution or if the partition should ``balance`` the labels across clients. 
- Download and partition the dataset by running the following command from the root dir: 
    - ```bash 
         poetry run python -m project.task.mnist_classification.dataset_preparation
        ```
- Specify which ``model_and_data``, ``train_structure``, and ``fit_config`` or ``eval_config`` to use in the ``conf/task/mnist.yaml file``. The defaults are a CNN, a simple classification training/testing loop, and configs controlling ``batch_size``, local client ``epochs``, and the ``learning_rate``. You can also specify which metrics to aggregate during fit/eval.
- Run the experiment using the following command from the root dir: 
    - ```bash 
        poetry run python -m project.main --config-name=mnist
        ```

Once a complete experiment has run, you can continue it for a specified number of epochs by running the following command from the root dir to change the output directory to the previous one. 
- ```bash 
    poetry run python -m project.main --config-name=mnist reuse_output_dir=<path_to_your_output_directory>
    ```
These are all the basic steps required to run a simple experiment. 

## Template Structure

The template uses poetry with the ``project`` name for the top-level package. All imports are made from this package, and no relative imports are allowed. The structure is as follows:

```
project
├── client
├── conf
├── dispatch
├── fed
├── main.py
├── task
├── types
└── utils
```
The the main packages of concern are:
- ``client``: Contains the client class, requires no changes
- ``conf``: This contains the Hydra configuration files specifying experiment behavior and the chosen ML task. 
- ``dispatch``: handles mapping a Hydra configuration to the ML task. 
- ``fed``: Contains the federated learning functionality such as client sampling and model parameter saving. Should require little to no modification.
- ``main``: a hydra entry point.
- ``task``: The ML task implementation includes the model, data loading, and training/testing. Almost all user changes should be made here. Tasks will typically include modules for the following:
    - ``dataset_preparation``: Hydra entry point which handles downloading the dataset and partitionin it. The partition can be generated on the fly during FL execution or saved into a partition directory with one folder per client containing train and test files---with the server test set being in the root directory of the partition dir. This needs to be executed prior to running the main experiment. It relies on the dataset part of the Hydra config.
    - ``dataset``: offers functionality to create the dataloaders for either the client fit/eval or for the centralized server evaluation.
    - ``dispatch`: Handles mapping the Hydra config to the required task configuration.
    - ``models``: Offers functionality to lazily create a model based on a received configuration.
    - ``train_test``: Offers functionality to train a model on a given dataset. This includes the effective train/test functions together with the config generation functions for the fit/eval stages of FL. The federated evaluation test function, if provided, should also be specified here.

Two tasks are already implemented:
- ``default``: A task providing generic functionality that may be reused across tasks. It requires no data and provides a minimum example of what a task must provide for the FL training to execute. 
- ``mnist_classification``: Uses the simple MNIST dataset with either a CNN or logistic regression model. 

> :warning: Prefer changing only the task module when possible.


## Enabling Pre-commit CI

To enable Continous Integration of your project via Pre-commit, all you need to do is allow pre-commit for a given repo from the [github marketplace](https://github.com/marketplace/pre-commit-ci/plan/MDIyOk1hcmtldHBsYWNlTGlzdGluZ1BsYW42MTI2#plan-6126). You should be aware that this is free only for public open-source repositories. 

## Long-term Public Projects

If you intend your project to run over multiple years and it does not require a private repository, prefer forking this codebase and creating your project as a branch. This will allow you to keep up to date with the latest changes to the template by syncing the main branch and merging it into your private branch.