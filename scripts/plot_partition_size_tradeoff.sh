#!/bin/bash

# ==============================================================================
# Script: plot_partition_size_tradeoff.sh
# Description: Sets up a Python virtual environment and runs the plotting script.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Configuration
VENV_DIR="$SCRIPT_DIR/.venv_plotting"
PYTHON_SCRIPT="$SCRIPT_DIR/plot_partition_size_tradeoff.py"
REQUIREMENTS="pandas matplotlib seaborn"

# 1. Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found."
    echo "Please ensure the python plotting script is saved as plot_partition_size_tradeoff.py in the same directory as this script."
    exit 1
fi

# 2. Check/Create Virtual Environment
if [ -d "$VENV_DIR" ]; then
    # echo "Found existing virtual environment in $VENV_DIR"
    :
else
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# 3. Activate Virtual Environment
source "$VENV_DIR/bin/activate"

# 4. Install Dependencies
echo "Installing dependencies ($REQUIREMENTS)..."
pip install -q --upgrade pip
pip install -q $REQUIREMENTS

# 5. Run the Plotting Script
echo "------------------------------------------------"
echo "Running $PYTHON_SCRIPT..."
python "$PYTHON_SCRIPT" "$@"

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "Success! Plots generated in the 'plots/' directory."
else
    echo "Error: Python script failed."
    exit 1
fi
