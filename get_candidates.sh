#!/bin/bash

#SBATCH --array=4-7
#SBATCH -p nvidia
#SBATCH -t 24:00:00
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

python -u get_candidates.py ${SLURM_ARRAY_TASK_ID} 3 25 10 5