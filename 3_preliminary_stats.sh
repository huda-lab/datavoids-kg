#!/bin/bash

#SBATCH --array=1-2
#SBATCH -p nvidia
#SBATCH -t 60:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o batch_jobs_out/%j.out

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --mail-user=jfg388@nyu.edu

# Activate any environments if required
module load miniconda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate datavoids


python 3_preliminary_stats.py --kg_name FB15k-237 --experiment_pairs_file experiment_pairs.json