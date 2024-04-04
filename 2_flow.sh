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
conda activate datavoids_env

python 2_flow.py --kg_name FB15k-237 --rels_to_test /film/actor/film./film/performance/film /film/director/film /tv/tv_producer/programs_produced./tv/tv_producer_term/program --num_heads_to_test 3 --num_attack_budget 25 --overlapping_budget_threshold 10 --diff_rankings 5