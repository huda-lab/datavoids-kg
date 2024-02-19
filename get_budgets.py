import json
import sys
import os
from Kelpie.dataset import Dataset
from helpers.helpers import print_fact
from helpers.knowledge_graph_simulation_experiment import (
    KnowledgeGraphMitigationExperiment,
)
from helpers.budget_helpers import get_good_bad_fact_budgets
from helpers.plotting_utilities import get_base_exp_name
from helpers.helpers import extract_subgraph_of_kg
# script args:
# 1. INPUT_FILE (e.g. experiment_inputs/input_1.txt)
# 2. DATASET_PATH (e.g. Kelpie_package/Kelpie/data/FB15k-237)
# 3. DATASET_NAME (e.g. FB15k-237)
# 4. BUDGET_FOLDER (folder where to store the budget file e.g. new_budgets)
# 5. NUM_ATTACK_BUDGET (how many budget facts to produce e.g. 25)
# 6. LABEL_MAP_PATH (optional, you can skip this argument)

INPUT_FILE = sys.argv[1]
good_fact, bad_fact = None, None
with open(INPUT_FILE, 'r') as in_file:
    good_fact, bad_fact = in_file.readlines()
    good_fact = good_fact.strip().split('-')
    bad_fact = bad_fact.strip().split('-')

print("Loading dataset...")
DATASET_PATH = sys.argv[2]
DATASET_NAME = sys.argv[3]
TRAIN_PATH = DATASET_PATH + "/train.txt"
TEST_PATH = DATASET_PATH + "/test.txt"
VALID_PATH = DATASET_PATH + "/valid.txt"
dataset = Dataset(
    name=DATASET_NAME,
    separator="\t",
    load=True,
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    valid_path=VALID_PATH,
)

if len(sys.argv) < 7:
    LABEL_MAP_PATH = None
    label_map = None
else:
    LABEL_MAP_PATH = sys.argv[6]
    label_map = json.load(open(LABEL_MAP_PATH, "r", encoding="utf-8"))


EXPERIMENT_NAME = get_base_exp_name(good_fact, bad_fact, label_map)
NUM_ATTACK_BUDGET = int(sys.argv[5])

reduced_dataset_path = f"reduced_datasets/{EXPERIMENT_NAME}_reduced_dataset.txt"
if not os.path.exists(reduced_dataset_path):
    print("Reduced dataset does not exist. Creating dataset")
    res = extract_subgraph_of_kg(
        dataset, [good_fact, bad_fact], 5, save_path=reduced_dataset_path
    )
reduced_dataset = Dataset(
    name=DATASET_NAME,
    separator="\t",
    load=True,
    train_path=reduced_dataset_path,
    test_path=TEST_PATH,
    valid_path=VALID_PATH,
)

BUDGET_FOLDER = sys.argv[4]
BUDGET_FILE = f"{BUDGET_FOLDER}/{EXPERIMENT_NAME}_budget.json"
get_good_bad_fact_budgets(
    reduced_dataset,
    good_fact,
    bad_fact,
    NUM_ATTACK_BUDGET,
    f"{BUDGET_FOLDER}/{EXPERIMENT_NAME}_budget_model.pt",
    BUDGET_FILE,
    "kelpie",
    "degree",
)
