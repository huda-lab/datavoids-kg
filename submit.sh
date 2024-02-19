#!/bin/bash

#SBATCH --array=1-2
#SBATCH -p nvidia
#SBATCH -t 60:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o batch_jobs_out/%j.out

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --mail-user=rh3015@nyu.edu

#activate any environments if required
module load miniconda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate opt-kelpie

python -u run_experiments.py experiment_inputs/input_${SLURM_ARRAY_TASK_ID}.txt