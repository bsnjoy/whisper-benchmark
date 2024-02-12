#!/bin/bash

# Check if the VIRTUAL_ENV environment variable is set
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Not inside a virtual environment. Attempting to activate one..."

    # Check if the virtual environment directory exists
    if [[ -d "venv" ]]; then
        # Activate the virtual environment
        . venv/bin/activate
        echo "Virtual environment activated."
    else
        echo "Virtual environment directory 'venv' not found."
    fi
else
    echo "Already inside a virtual environment ($VIRTUAL_ENV)."
fi

python3 benchmark.py
