#!/bin/bash
#SBATCH --job-name=test_experiment_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=a.kraljevic.2000@hotmail.com
#SBATCH --mail-type="ALL"
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1

# Source the Python virtual environment
source $HOME/quantumrl/python_venv/bin/activate
TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"

echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"

echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

echo "## Running test"
python experiment_alice.py

echo "## Test finished. Goodbye"