#!/bin/bash

#SBATCH -p nvidia
#SBATCH -t 72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o batch_jobs_out/%j.out


#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --mail-user=jfg388@nyu.edu

#activate any environments if required
module load miniconda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jr-env

echo "SLURM_JOB_ID:        $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID:  $SLURM_ARRAY_TASK_ID"

python yago310_get_new_budget_files.py