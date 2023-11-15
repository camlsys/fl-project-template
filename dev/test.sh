#!/bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
echo $SCRIPT_DIR
cd $SCRIPT_DIR/..
echo "Current directory: $(pwd)"
PROJECT_NAME="project"

echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
poetry run python -m isort --check-only $PROJECT_NAME/
echo "- isort: done"

echo "- black: start"
poetry run python -m black --check $PROJECT_NAME/
echo "- black: done"

echo "- docformatter: start"
poetry run python -m docformatter -c -r $PROJECT_NAME/
echo "- docformatter:  done"

echo "- ruff: start"
poetry run python -m ruff check $PROJECT_NAME/
echo "- ruff: done"

echo "- mypy: start"
poetry run python -m mypy --incremental --show-traceback $PROJECT_NAME/
echo "- mypy: done"


