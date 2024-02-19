#!/bin/bash

#SBATCH -p nvidia
#SBATCH -t 72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o batch_jobs_out/%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jfg388@nyu.edu

# Activate any environments if required
module load miniconda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jr-env

echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Define your tuples as strings
# TUPLES=(
#   "('friedrich_hayek', 'iscitizenof', 'united_kingdom'),('friedrich_hayek', 'iscitizenof', 'austria')"
#   "('john_burridge', 'isaffiliatedto', 'manchester_city_f.c.'),('john_burridge', 'isaffiliatedto', 'sheffield_united_f.c.')"
#   # "('franz_kafka', 'iscitizenof', 'austria-hungary'),('franz_kafka', 'iscitizenof', 'czechoslovakia')"
# )
# TUPLES=(
#   "('adam_smith', 'iscitizenof', 'scotland'),('adam_smith', 'iscitizenof', 'kingdom_of_great_britain')"
#   "('china', 'exports', 'wordnet_fabric_103309808'),('china', 'exports', 'wordnet_apparel_102728440')"
#   "('honduras', 'exports', 'wordnet_banana_112352287'),('honduras', 'exports', 'wordnet_coffee_bean_107929351')"
# )
# TUPLES=(
#   "('friedrich_hayek', 'iscitizenof', 'united_kingdom'),('friedrich_hayek', 'iscitizenof', 'austria')"
#   "('china', 'exports', 'wordnet_fabric_103309808'),('china', 'exports', 'wordnet_apparel_102728440')"
#   "('john_burridge', 'isaffiliatedto', 'manchester_city_f.c.'),('john_burridge', 'isaffiliatedto', 'sheffield_united_f.c.')"
# )
TUPLES=(
  "('friedrich_hayek', 'iscitizenof', 'united_kingdom'),('friedrich_hayek', 'iscitizenof', 'austria')"
  "('china', 'exports', 'wordnet_fabric_103309808'),('china', 'exports', 'wordnet_apparel_102728440')"
)

STRATEGIES_1=("multi_greedy" "approx_greedy" "random")
STRATEGIES_2=("multi_greedy" "approx_greedy" "random")

# Calculate the number of strategies in each array
NUM_STRATEGIES_1=${#STRATEGIES_1[@]}
NUM_STRATEGIES_2=${#STRATEGIES_2[@]}
NUM_TUPLES=${#TUPLES[@]}

# Calculate tuple and strategy indices based on SLURM_ARRAY_TASK_ID
TUPLE_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_STRATEGIES_1 * NUM_STRATEGIES_2) % NUM_TUPLES))
MITIGATOR_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_STRATEGIES_2 % NUM_STRATEGIES_1))
DISINFORMER_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_STRATEGIES_2))

# Pass the tuple and strategy to the Python script
python -u yago310_run_experiments.py --tuple "${TUPLES[$TUPLE_INDEX]}" --mitigator_strategy "${STRATEGIES_1[$MITIGATOR_INDEX]}" --disinformer_strategy "${STRATEGIES_2[$DISINFORMER_INDEX]}"

