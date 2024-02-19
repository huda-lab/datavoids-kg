#!/bin/bash

#SBATCH --array=1,2,3,4,5,6
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

python -u yago310_get_candidates.py ${SLURM_ARRAY_TASK_ID} 3 25 10 5 2