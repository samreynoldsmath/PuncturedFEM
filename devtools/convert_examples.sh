#!/bin/bash
# Converts all Jupyter notebooks in examples/ to Python scripts

# directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# directory containing Jupyter notebooks
NOTEBOOK_DIR="$SCRIPT_DIR/../examples"

# clear outputs
jupyter nbconvert --clear-output --inplace "$NOTEBOOK_DIR"/*.ipynb

# convert notebooks to python scripts
jupyter nbconvert --to script \
	--output-dir="$NOTEBOOK_DIR" "$NOTEBOOK_DIR"/*.ipynb