#!/bin/bash

#SBATCH --array=10
#SBATCH -p nvidia
#SBATCH -t 1:00:00
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

python -u get_budgets.py experiment_inputs/input_${SLURM_ARRAY_TASK_ID}.txt "Kelpie_package/Kelpie/data/FB15k-237" "FB15k-237" "new_budgets" 25 "entity2wikidata.json"