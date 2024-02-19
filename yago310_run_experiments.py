"""

python3 yago310_run_experiments.py --tuple "('friedrich_hayek', 'iscitizenof', 'united_kingdom'),('friedrich_hayek', 'iscitizenof', 'austria')"

"""

import json
import traceback
import os
import sys
import ast
import argparse
from Kelpie.dataset import Dataset
from helpers.helpers import extract_subgraph_of_kg, print_fact
from helpers.knowledge_graph_simulation_experiment import (
    KnowledgeGraphMitigationExperiment,
)
from helpers.constants import SEED
from helpers.budget_helpers import get_good_bad_fact_budgets

DATASET_NAME = "yago310"
DATASET_PATH = 'Kelpie_package/Kelpie/data/YAGO3-10'
TRAIN_PATH = DATASET_PATH + '/train.txt'
TEST_PATH = DATASET_PATH + '/test.txt'
VALID_PATH = DATASET_PATH + '/valid.txt'
dataset = Dataset(name=DATASET_NAME,
                  load=True,
                  train_path=TRAIN_PATH,
                  test_path=TEST_PATH,
                  valid_path=VALID_PATH)
train_test_valid_paths = [TRAIN_PATH, TEST_PATH, VALID_PATH]
EXP_FOLDER = "results/yago310/new_mo_strategy_4"


# strategies_1 = ["random", "greedy", "multi_greedy"]
# strategies_2 = ["multi_greedy", "greedy", "random" ]

os.makedirs("reduced_datasets", exist_ok=True)

# tuples_to_experiment_with = [
#     [
#         ('friedrich_hayek', 'iscitizenof', 'united_kingdom'),
#         ('friedrich_hayek', 'iscitizenof', 'austria')
#     ],
#     [
#         ('john_burridge', 'isaffiliatedto', 'manchester_city_f.c.'),
#         ('john_burridge', 'isaffiliatedto', 'sheffield_united_f.c.')
#     ],
#     [
#         ('franz_kafka', 'iscitizenof', 'austria-hungary'),
#         ('franz_kafka', 'iscitizenof', 'czechoslovakia')
#     ],
# ]



def main(tuple_str, mitigator_strategy, disinformer_strategy):
    good_fact, bad_fact = ast.literal_eval(tuple_str)

    if not good_fact or not bad_fact:
        raise ValueError("Issue reading input")

    print("Now testing")
    print(good_fact)
    print("versus")
    print(bad_fact)


    EXPERIMENT_NAME = f"{good_fact}_{bad_fact}"
    print("experiment name", EXPERIMENT_NAME)
    NUM_RANDOM_REPS = 5
    NUM_ATTACK_BUDGET = 25
    REMOVE_OVERLAPPING_BUDGET_FROM_DV = True
    BUDGET_FILE = f"budgets/{EXPERIMENT_NAME}_budget.json"

    reduced_dataset_path = f"reduced_datasets/yago310/{EXPERIMENT_NAME}.txt"
    if not os.path.exists(reduced_dataset_path):
        print("Reduced dataset does not exist. Creating dataset")
        res = extract_subgraph_of_kg(
            dataset, [good_fact, bad_fact], 5, save_path=reduced_dataset_path
        )
    reduced_dataset = Dataset(name=DATASET_NAME,
                  load=True,
                  train_path=reduced_dataset_path,
                  test_path=TEST_PATH,
                  valid_path=VALID_PATH)
    
    if not os.path.exists(BUDGET_FILE):
        print("Budget does not exist. Creating budget.")
        # get_good_bad_fact_budgets(dataset: Dataset, good_fact: tuple, bad_fact: tuple, num_budget: int, trained_model_saving_path: str, budget_dump_file: str, budget_strategy: str) -> dict:
        get_good_bad_fact_budgets(
        reduced_dataset,
        good_fact,
        bad_fact,
        NUM_ATTACK_BUDGET,
        f"results/yago310/{EXPERIMENT_NAME}_budget_r_model.pt",
        BUDGET_FILE,
        "kelpie",
        "degree",
    )

    good_entity = good_fact[2]
    bad_entity = bad_fact[2]
    print("good entity:", good_entity)
    print("bad entity:", bad_entity)
    print(f"Running {mitigator_strategy}-{disinformer_strategy} experiment")
    try:
        experiment_results_folder = f"{EXP_FOLDER}/{EXPERIMENT_NAME}_{mitigator_strategy}_{disinformer_strategy}"
        experiment = KnowledgeGraphMitigationExperiment(
            experiment_results_folder,
            DATASET_NAME,
            None,
            good_fact,
            good_entity,
            bad_entity,
            og_train_test_valid_paths=[reduced_dataset_path, TEST_PATH, VALID_PATH],
            prediction_type="tail",
            reduce_original=False,
            mode="necessary",
            model_name="ComplEx",
            mitigator_strategy=mitigator_strategy,
            disinformer_strategy=disinformer_strategy,
            num_attack_budget=NUM_ATTACK_BUDGET,
            num_experiments_random=NUM_RANDOM_REPS,
            base_random_strategy_seed=SEED,
            lp_model_seed=SEED,
            remove_overlapping_budget_from_dv=REMOVE_OVERLAPPING_BUDGET_FROM_DV,
            save_dataset_each_round=False,
            budget_file=BUDGET_FILE,
            resume_experiment=True,
        )
        experiment.run_experiment()
    except Exception as e:
        print("Exception: ", e)
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuple", required=True, help="String representation of a fact tuple.")
    parser.add_argument("--mitigator_strategy", required=True, help="Mitigator strategy to use.")
    parser.add_argument("--disinformer_strategy", required=True, help="Disinformer strategy to use.")
    args = parser.parse_args()

    main(args.tuple, args.mitigator_strategy, args.disinformer_strategy)
