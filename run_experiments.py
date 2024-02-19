import json
import os
import sys
from Kelpie.dataset import Dataset
from helpers.helpers import extract_subgraph_of_kg, print_fact
from helpers.knowledge_graph_simulation_experiment import (
    KnowledgeGraphMitigationExperiment,
)
from helpers.constants import SEED
from helpers.budget_helpers import get_good_bad_fact_budgets

FB15K237_DATASET_PATH = "Kelpie_package/Kelpie/data/FB15k-237"
TRAIN_PATH = FB15K237_DATASET_PATH + "/train.txt"
TEST_PATH = FB15K237_DATASET_PATH + "/test.txt"
VALID_PATH = FB15K237_DATASET_PATH + "/valid.txt"
FB15K237_DATASET = Dataset(
    name="FB15k-237",
    load=True,
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    valid_path=VALID_PATH,
)

LABEL_MAP_PATH = "entity2wikidata.json"
LABEL_MAP = json.load(open(LABEL_MAP_PATH))

STRATEGIES = ["random", "greedy", "multi_greedy"]
# STRATEGIES = ['random', 'greedy', 'approx_greedy', 'neighbor', 'multi_greedy']

os.makedirs("reduced_datasets", exist_ok=True)

INPUT_FILE = sys.argv[1]

with open(INPUT_FILE, "r") as f:
    lines = f.readlines()
    good_fact = lines[0].strip().split("-")
    bad_fact = lines[1].strip().split("-")

if not good_fact or not bad_fact:
    raise ValueError("Issue reading input")

print("Now testing")
print_fact(good_fact, LABEL_MAP)
print("versus")
print_fact(bad_fact, LABEL_MAP)


EXPERIMENT_NAME = f"{LABEL_MAP[good_fact[0]]['label']}_{LABEL_MAP[good_fact[2]]['label']}_{LABEL_MAP[bad_fact[2]]['label']}"
NUM_RANDOM_REPS = 5
NUM_ATTACK_BUDGET = 25
REMOVE_OVERLAPPING_BUDGET_FROM_DV = True
EXP_FOLDER = "final_paper_results"
BUDGET_FILE = f"new_budgets/{EXPERIMENT_NAME}_budget.json"

reduced_dataset_path = f"reduced_datasets/{EXPERIMENT_NAME}_reduced_dataset.txt"
if not os.path.exists(reduced_dataset_path):
    print("Reduced dataset does not exist. Creating dataset")
    res = extract_subgraph_of_kg(
        FB15K237_DATASET, [good_fact, bad_fact], 5, save_path=reduced_dataset_path
    )
reduced_dataset = Dataset(
    name="FB15K-237",
    separator="\t",
    load=True,
    train_path=reduced_dataset_path,
    test_path=TEST_PATH,
    valid_path=VALID_PATH,
)

if not os.path.exists(BUDGET_FILE):
    print("Budget does not exist. Creating budget.")
    get_good_bad_fact_budgets(
        reduced_dataset,
        good_fact,
        bad_fact,
        NUM_ATTACK_BUDGET,
        f"results/{EXP_FOLDER}/{EXPERIMENT_NAME}_budget_model.pt",
        BUDGET_FILE,
        "kelpie",
        "degree",
    )

good_entity = good_fact[2]
bad_entity = bad_fact[2]
print("good entity:", LABEL_MAP[good_entity]["label"])
print("bad entity:", LABEL_MAP[bad_entity]["label"])

for mitigator_strategy in STRATEGIES:
    for disinformer_strategy in STRATEGIES:
        print(f"Running {mitigator_strategy}-{disinformer_strategy} experiment")
        try:
            experiment = KnowledgeGraphMitigationExperiment(
                f"{EXP_FOLDER}/{EXPERIMENT_NAME}_{mitigator_strategy}_{disinformer_strategy}",
                "FB15k-237",
                LABEL_MAP_PATH,
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
                remove_overlapping_budget_from_dv= True,
                save_dataset_each_round=False,
                budget_file=BUDGET_FILE,
                resume_experiment=False,
                cost_type="degree",
            )
            experiment.run_experiment()
        except Exception as e:
            print("Exception: ", e)

# Random-random
# print("Running Random-Random experiment")
# experiment_random_random_1 = KnowledgeGraphMitigationExperiment(
#     f"{EXP_FOLDER}/{EXPERIMENT_NAME}_random_random_1",
#     "FB15k-237",
#     LABEL_MAP_PATH,
#     good_fact,
#     good_entity,
#     bad_entity,
#     og_train_test_valid_paths=[reduced_dataset_path, TEST_PATH, VALID_PATH],
#     prediction_type="tail",
#     reduce_original=False,
#     mode="necessary",
#     model_name="ComplEx",
#     mitigator_strategy="random",
#     disinformer_strategy="random",
#     num_attack_budget=NUM_ATTACK_BUDGET,
#     num_experiments_random=NUM_RANDOM_REPS,
#     base_random_strategy_seed=None,
#     lp_model_seed=SEED,
#     remove_overlapping_budget_from_dv=REMOVE_OVERLAPPING_BUDGET_FROM_DV,
#     save_dataset_each_round=False,
#     budget_file=BUDGET_FILE,
#     resume_experiment=True,
# )

# experiment_random_random_1.run_experiment()

# experiment_random_random_2 = KnowledgeGraphMitigationExperiment(
#     f"{EXP_FOLDER}/{EXPERIMENT_NAME}_random_random_2",
#     "FB15k-237",
#     LABEL_MAP_PATH,
#     good_fact,
#     good_entity,
#     bad_entity,
#     og_train_test_valid_paths=[reduced_dataset_path, TEST_PATH, VALID_PATH],
#     prediction_type="tail",
#     reduce_original=False,
#     mode="necessary",
#     model_name="ComplEx",
#     mitigator_strategy="random",
#     disinformer_strategy="random",
#     num_attack_budget=NUM_ATTACK_BUDGET,
#     num_experiments_random=NUM_RANDOM_REPS,
#     base_random_strategy_seed=None,
#     lp_model_seed=SEED,
#     remove_overlapping_budget_from_dv=REMOVE_OVERLAPPING_BUDGET_FROM_DV,
#     save_dataset_each_round=False,
#     resume_experiment=True,
# )
# experiment_random_random_2.run_experiment()
