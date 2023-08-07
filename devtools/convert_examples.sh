#!/bin/bash

notebook_dir="../examples"
jupyter nbconvert --to script --comment all \
	--output-dir="$notebook_dir" "$notebook_dir"/*.ipynb
