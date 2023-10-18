#!/bin/bash
#SBATCH --job-name=experiment-lr_0.01_0.001-dc_0.001
#SBATCH --output=%x_%j.out
#SBATCH --mail-user=a.kraljevic.2000@hotmail.com
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-medium"
#SBATCH --time=07:30:00
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=3

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load Python/3.10.8-GCCcore-12.2.0

# Source the Python virtual environment
source $HOME/quantumrl/python_venv/bin/activate

echo "[$SHELL] #### Starting Python experiment"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python experiment_alice.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"