#!/bin/bash

# Define your tuples and strategies here or source them from another file
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


# Calculate total number of jobs
NUM_TUPLES=${#TUPLES[@]}
NUM_STRATEGIES_1=${#STRATEGIES_1[@]}
NUM_STRATEGIES_2=${#STRATEGIES_2[@]}
TOTAL_JOBS=$((NUM_TUPLES * NUM_STRATEGIES_1 * NUM_STRATEGIES_2))

# Print the total number of jobs
echo "Total number of jobs to be submitted: $TOTAL_JOBS"

# Submit the job with the correct array range
sbatch --array=0-$((TOTAL_JOBS - 1)) yago310_run_experiments.sh
