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

# ready to run
# python -u 3_flow.py \
#     --kg_name "FB15k-237" \
#     --good_fact "/m/014zcr-/film/actor/film./film/performance/film-/m/07w8fz" \
#     --bad_fact "/m/014zcr-/film/actor/film./film/performance/film-/m/0418wg" \
#     --part 0 \
#     --num_attack_budget 25 \
#     --num_random_reps 10

# Activate any environments if required
module load miniconda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate datavoids

# George Clooney actor Good Night, and Good Luck. Ocean’s Twelve (input_7.txt) (did appear in flow 2)

# Set the input parameters
KG_NAME="FB15k-237"
GOOD_FACT="/m/014zcr-/film/actor/film./film/performance/film-/m/0418wg"
BAD_FACT="/m/014zcr-/film/actor/film./film/performance/film-/m/07w8fz"
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
