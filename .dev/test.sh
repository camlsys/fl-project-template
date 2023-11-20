#!/bin/bash

echo "=== test.sh ==="

echo "- Start Python checks"

echo "- isort: start"
poetry run python -m isort --check-only project/
echo "- isort: done"

echo "- black: start"
poetry run python -m black --check project/
echo "- black: done"

echo "- docformatter: start"
poetry run python -m docformatter -c -r project/
echo "- docformatter:  done"

echo "- ruff: start"
poetry run python -m ruff check project/
echo "- ruff: done"

echo "- mypy: start"
poetry run python -m mypy --incremental --show-traceback project/
echo "- mypy: done"