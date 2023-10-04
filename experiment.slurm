#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="a.kraljevic.2000@hotmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --partition="cpu-short"
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --mem=10M