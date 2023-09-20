#!/bin/bash
# Converts all Jupyter notebooks in examples/ to Python scripts

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
NOTEBOOK_DIR="$SCRIPT_DIR/../examples"
jupyter nbconvert --to script --comment all \
	--output-dir="$NOTEBOOK_DIR" "$NOTEBOOK_DIR"/*.ipynb