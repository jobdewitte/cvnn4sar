#!/bin/bash

conda init bash
source ~/.bashrc 

ENV_YAML=environment.yaml
TARGET_DIR=../CVNN/pytorch-complex

# Check if the YAML file exists
if [ ! -f "$ENV_YAML" ]; then
    echo "Error: Environment YAML file '$ENV_YAML' does not exist."
    exit 1
fi

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Target directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Install the conda environment from the YAML file
echo "Installing conda environment from '$ENV_YAML'..."
conda env create -f "$ENV_YAML"

# Extract the environment name from the YAML file
ENV_NAME=$(head -n 1 "$ENV_YAML" | cut -d ' ' -f 2)

# Activate the newly created conda environment
echo "Activating conda environment '$ENV_NAME'..."

conda activate "$ENV_NAME"

# Change to the target directory
echo "Changing to directory '$TARGET_DIR'..."
cd "$TARGET_DIR"

# Run pip install .
echo "Running 'pip install .'..."
pip install .

# Deactivate the conda environment
echo "Deactivating conda environment..."
conda deactivate

echo "Script completed successfully."