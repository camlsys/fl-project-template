#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
echo $SCRIPT_DIR
cd $SCRIPT_DIR/..
echo "Current directory: $(pwd)"
PROJECT_NAME="project"

echo "Formatting started"
echo "Run isort"
poetry run python -m isort $PROJECT_NAME/
echo "Run black"
poetry run python -m black -q $PROJECT_NAME/
echo "Run yamlfix"
poetry run yamlfix $PROJECT_NAME/conf/
poetry run yamlfix sweepers/
echo "Run docformatter"
poetry run python -m docformatter -i -r $PROJECT_NAME/
echo "Run ruff"
poetry run python -m ruff check --fix $PROJECT_NAME/
echo "Formatting done"
