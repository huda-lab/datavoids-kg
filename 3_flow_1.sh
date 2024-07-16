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

# python -u 3_flow.py \
#     --kg_name "FB15k-237" \
#     --good_fact "/m/0151w_-/film/director/film-/m/0h03fhx" \
#     --bad_fact "/m/0151w_-/film/director/film-/m/07kh6f3" \
#     --part 0 \
#     --num_attack_budget 25 \
#     --num_random_reps 10 \
#     --regenerate_files


# Ben Affleck director Argo The Town (input_1.txt) 


# Activate any environments if required
module load miniconda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate datavoids

# Set the input parameters
KG_NAME="FB15k-237"
GOOD_FACT="/m/0151w_-/film/director/film-/m/0h03fhx"
BAD_FACT="/m/0151w_-/film/director/film-/m/07kh6f3"
NUM_ATTACK_BUDGET=25  
NUM_RANDOM_REPS=10  
PART=${SLURM_ARRAY_TASK_ID}

# Run the Python script with the specified arguments
python -u 3_flow.py \
    --kg_name $KG_NAME \
    --good_fact $GOOD_FACT \
    --bad_fact $BAD_FACT \
    --part $PART \
    --num_attack_budget $NUM_ATTACK_BUDGET \
    --num_random_reps $NUM_RANDOM_REPS \
    --regenerate_files
