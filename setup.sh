#!/bin/bash

VPOETRY_HOME=""
VPYENV_ROOT=""
VPYTHON_VERSION="3.11.6"

if [ -z "$(command -v poetry)" ]; then

  if [ -z "$VPOETRY_HOME" ]; then
  echo "POETRY_HOME is empty, please add it to the script or input a home now."
  read VPOETRY_HOME
  else
    echo "POETRY_HOME is not empty"
  fi

  mkdir -p $VPOETRY_HOME
  curl -sSL https://install.python-poetry.org | POETRY_HOME=$VPOETRY_HOME python3 -
else
  echo "Poetry is already installed"
fi

#!/bin/bash

if ! [ -x "$(command -v pyenv)" ]; then
  if [ -z "$VPYENV_ROOT" ]; then
  echo "PYENV_ROOT is empty, please add it to the script or input a home now."
  read VPYENV_ROOT
  else
    echo "PYENV_ROOT is not empty"
  fi


  curl https://pyenv.run | bash
  eval "$(PYENV_ROOT=$VPYENV_ROOT pyenv init -)"
  echo "export PYENV_ROOT=\"$VPYENV_ROOT\"" >> ~/.bashrc
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
  echo 'eval "$(pyenv init -)"' >> ~/.bashrc
  echo "export PYENV_ROOT=\"$VPYENV_ROOT\"" >> ~/.profile
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
  echo 'eval "$(pyenv init -)"' >> ~/.profile
else
  echo "Pyenv is already installed"
fi

if pyenv versions | grep -q $VPYTHON_VERSION; then
  echo "Python $VPYTHON_VERSION is already installed"
else
  pyenv install $VPYTHON_VERSION
fi


current_version=$(pyenv local)

if [ "$current_version" == "$VPYTHON_VERSION" ]; then
  echo "The local Python version is already set to $VPYTHON_VERSION"
else
  pyenv local $VPYTHON_VERSION
fi
poetry env use $VPYTHON_VERSION

poetry install

poetry run pre-commit install 

poetry run pre-commit run --all-files --hook-stage push
