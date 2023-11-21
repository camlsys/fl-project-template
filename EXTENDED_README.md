
# Extended Readme

> Scientific projects are expected to run in a machine with Ubuntu 22.04

While `README.md` should include information about the project you implement and how to run it, this _extended_ readme provides more generally the instructions to follow before your work can be run on the CaMLSys cluster or used collaboratively. Please follow closely these instructions. It is likely that you have already completed steps 1-2.

1. Click use this template on the Github Page.
2. Run ```./setup.sh``` to set up the template project.
3. Add your additional dependencies to the `pyproject.toml` (see below a few examples on how to do it). Read more about Poetry below in this `EXTENDED_README.md`.
4. Regularly check that your coding style and the documentation you add follow good coding practices. To test whether your code meets the requirements, please run ``./setup.sh`` or
    ```bash
    poetry run pre-commit install 
    ```
5. If you update the `.pre-commit-config.yaml` file or change the tools in the toml file you should run the following. Feel free to run this whenever you need a sanity check that the entire codebase is up to standard.
    ```bash
    poetry run pre-commit run --all-files --hook-stage push
    ```
6. Ensure that the Python environment for your project can be created without errors by simply running `poetry install` and that this is properly described later when you complete the `Setup` section in `README.md`. This is specially important if your environment requires additional steps after doing `poetry install` or ```./setup.sh```.
7. Ensure that your project runs with default arguments by running `poetry run python -m project.main`. Then, describe this and other forms of running your code in the `Using the Project` section in `README.md`.
8. Once your code is ready and you have checked:
    *    that following the instructions in your `README.md` the Python environment can be created correctly

    *    that running the code following your instructions can reproduce the experiments you setup for your paper
   
   ,then you just need to invite collaborators to your project if this is in a private repo or publish it to the camlsys github if it is meant as an artefact. If you feel like some improvements can be shared across projects please open a PR for the template. 

> Once you are happy with your project please delete this `EXTENDED_README.md` file.


## About Poetry

We use Poetry to manage the Python environment for each individual project. You can follow the instructions [here](https://python-poetry.org/docs/) to install Poetry in your machine. The ``./setup.sh`` script already handles all of these steps for you, however, you need to understand how they work to properly change the template.


### Specifying a Python Version (optional)
By default, Poetry will use the Python version in your system. In some settings, you might want to specify a particular version of Python to use inside your Poetry environment. You can do so with [`pyenv`](https://github.com/pyenv/pyenv). Check the documentation for the different ways of installing `pyenv`, but one easy way is using the [automatic installer](https://github.com/pyenv/pyenv-installer):
```bash
curl https://pyenv.run | bash # then, don't forget links to your .bashrc/.zshrc
```

You can then install any Python version with `pyenv install <python-version>` (e.g. `pyenv install 3.9.17`). Then, in order to use that version for your project, you'd do the following:

```bash
# cd to your project directory (i.e. where the `pyproject.toml` is)
pyenv local <python-version>

# set that version for poetry
poetry env use <python-version>

# then you can install your Poetry environment (see the next setp)
```

### Installing Your Environment
With the Poetry tool already installed, you can create an environment for this project with commands:
```bash
# run this from the same directory as the `pyproject.toml` file is
poetry install
```

This will create a basic Python environment with just Flower and additional packages, including those needed for simulation. Next, you should add the dependencies for your code. It is **critical** that you fix the version of the packages you use using a `=` not a `=^`. You can do so via [`poetry add`](https://python-poetry.org/docs/cli/#add). Below are some examples:

```bash
# For instance, if you want to install tqdm
poetry add tqdm==4.65.0

# If you already have a requirements.txt, you can add all those packages (but ensure you have fixed the version) in one go as follows:
poetry add $( cat requirements.txt )
```
With each `poetry add` command, the `pyproject.toml` gets automatically updated so you don't need to keep that `requirements.txt` as part of this project.


More critically however, is adding your ML framework of choice to the list of dependencies. For some frameworks you might be able to do so with the `poetry add` command. Check [the Poetry documentation](https://python-poetry.org/docs/cli/#add) for how to add packages in various ways. For instance, let's say you want to use PyTorch:

```bash
# with plain `pip`  you'd run a command such as:
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# to add the same 3 dependencies to your Poetry environment you'd need to add the URL to the wheel that the above pip command auto-resolves for you.
# You can find those wheels in `https://download.pytorch.org/whl/cu117`. Copy the link and paste it after the `poetry add` command.
# For instance to add `torch==1.13.1+cu117` and a x86 Linux system with Python3.8 you'd:
poetry add https://download.pytorch.org/whl/cu117/torch-1.13.1%2Bcu117-cp38-cp38-linux_x86_64.whl
# you'll need to repeat this for both `torchvision` and `torchaudio`
```
The above is just an example of how you can add these dependencies. Please refer to the Poetry documentation to extra reference.

If all attempts fail, you can still install packages via standard `pip`. You'd first need to source/activate your Poetry environment.
```bash
# first ensure you have created your environment
# and installed the base packages provided in the template
poetry install 

# then activate it
poetry shell
```
Now you are inside your environment (pretty much as when you use `virtualenv` or `conda`) so you can install further packages with `pip`. Please note that, unlike with `poetry add`, these extra requirements won't be captured by `pyproject.toml`. Therefore, please ensure that you provide all instructions needed to: (1) create the base environment with Poetry and (2) install any additional dependencies via `pip` when you complete your `README.md`.