#!/bin/bash

#SBATCH -p nvidia
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=8
#SBATCH -o batch_jobs_out/%j.out

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --mail-user=rh3015@nyu.edu

#activate any environments if required
module load miniconda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate opt-kelpie

python -u plot.py