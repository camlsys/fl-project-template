#!/bin/bash
USER_NAME=$(echo $USER)
VPOETRY_HOME="/nfs-share/$USER_NAME/.poetry"
VPYENV_ROOT="/nfs-share/$USER_NAME/.pyenv"
VPYTHON_VERSION="3.11.6"

if ! [ -x "$(command -v pyenv)" ]; then
  if [ -z "$VPYENV_ROOT" ]; then
  echo "PYENV_ROOT is empty, please add it to the script or input a home now."
  read VPYENV_ROOT
  else
    echo "PYENV_ROOT is not empty"
  fi

  curl https://pyenv.run | bash
  PYENV_ROOT=$VPYENV_ROOT
  PYENV_BIN=$PYENV_ROOT/bin/pyenv

  eval "$($PYENV_BIN init -)"
  echo "export PYENV_ROOT=\"$VPYENV_ROOT\"" >> /home/$USER_NAME/.bashrc
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /home/$USER_NAME/.bashrc
  echo 'eval "$(pyenv init -)"' >> /home/$USER_NAME/.bashrc
  echo "export PYENV_ROOT=\"$VPYENV_ROOT\"" >> /home/$USER_NAME/.profile
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /home/$USER_NAME/.profile
  echo 'eval "$(pyenv init -)"' >> /home/$USER_NAME/.profile
else
  echo "Pyenv is already installed"
fi

if [ -z "$(command -v poetry)" ]; then

  if [ -z "$VPOETRY_HOME" ]; then
  echo "POETRY_HOME is empty, please add it to the script or input a home now."
  read VPOETRY_HOME
  else
    echo "POETRY_HOME is not empty"
  fi

  mkdir -p $VPOETRY_HOME
  curl -sSL https://install.python-poetry.org | POETRY_HOME=$VPOETRY_HOME python3 -
  
  POETRY_BIN="$VPOETRY_HOME/bin/poetry"

  echo 'command -v poetry >/dev/null || export PATH="'$VPOETRY_HOME'/bin:$PATH"' >> /home/$USER_NAME/.bashrc
  echo 'command -v poetry >/dev/null || export PATH="'$VPOETRY_HOME'/bin:$PATH"' >> /home/$USER_NAME/.profile

else
  echo "Poetry is already installed"
fi

if $PYENV_BIN versions | grep -q $VPYTHON_VERSION; then
  echo "Python $VPYTHON_VERSION is already installed"
else
  $PYENV_BIN install $VPYTHON_VERSION
fi


current_version=$($PYENV_BIN local)

if [ "$current_version" = "$VPYTHON_VERSION" ]; then
  echo "The local Python version is already set to $VPYTHON_VERSION"
else
  $PYENV_BIN local $VPYTHON_VERSION
fi
$POETRY_BIN env use $VPYTHON_VERSION

$POETRY_BIN install

# Install pre-commit hooks
$POETRY_BIN run pre-commit install

# Command to run all pre-commit hooks
$POETRY_BIN run pre-commit run --all-files --hook-stage push
