import json
import sys
import pandas as pd
from Kelpie.dataset import Dataset
from helpers.plotting_utilities import get_base_exp_name
from helpers.helpers import get_readable_fact_str

experiment_pairs = []
# for i in range(1):
with open(f'experiment_inputs/input_10.txt', 'r') as in_file:
    good_fact, bad_fact = in_file.readlines()
    good_fact = good_fact.strip().split('-')
    bad_fact = bad_fact.strip().split('-')
    experiment_pairs.append((good_fact, bad_fact))

# script args:
# 1. DATASET_PATH (e.g. Kelpie_package/Kelpie/data/FB15k-237)
# 2. DATASET_NAME (e.g. FB15k-237)
# 3. BUDGET_FOLDER (folder where to find the budget files e.g. new_budgets)
# 4. SAVE_FOLDER (folder where to save the statistics about each fact e.g. budget_stats)
# 5. LABEL_MAP_PATH (optional, you can skip this argument)

print("Loading dataset...")
DATASET_PATH = sys.argv[1]
DATASET_NAME = sys.argv[2]
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

BUDGET_FOLDER = sys.argv[3]
SAVE_FOLDER = sys.argv[4]

if len(sys.argv) < 6:
    LABEL_MAP_PATH = None
    label_map = None
else:
    LABEL_MAP_PATH = sys.argv[5]
    label_map = json.load(open(LABEL_MAP_PATH, "r", encoding="utf-8"))


for good_fact, bad_fact in experiment_pairs:
    budget_df = pd.DataFrame(
        columns=["claim", "budget_degrees", "budget_relevance", "budget_costs"]
    )
    exp_name = get_base_exp_name(good_fact, bad_fact, label_map)
    with open(f"{BUDGET_FOLDER}/{exp_name}_budget.json", "r") as inf:
        budget = json.load(inf)
        good_budget_stats = {}
        good_budget_stats["budget_degrees"] = {
            get_readable_fact_str(k.split("-"), label_map): v
            for k, v in budget["good_budget_degrees"].items()
        }
        good_budget_stats["budget_relevance"] = {
            get_readable_fact_str(k.split("-"), label_map): v
            for k, v in budget["good_budget_relevance"].items()
        }
        good_budget_stats["budget_costs"] = {
            get_readable_fact_str(k.split("-"), label_map): v
            for k, v in budget["good_budget_costs"].items()
        }
        good_budget_df = pd.DataFrame.from_dict(good_budget_stats, orient="columns")
        good_budget_df["claim"] = get_readable_fact_str(good_fact, label_map)

        bad_budget_stats = {}
        bad_budget_stats["budget_degrees"] = {
            get_readable_fact_str(k.split("-"), label_map): v
            for k, v in budget["bad_budget_degrees"].items()
        }
        bad_budget_stats["budget_relevance"] = {
            get_readable_fact_str(k.split("-"), label_map): v
            for k, v in budget["bad_budget_relevance"].items()
        }
        bad_budget_stats["budget_costs"] = {
            get_readable_fact_str(k.split("-"), label_map): v
            for k, v in budget["bad_budget_costs"].items()
        }
        bad_budget_df = pd.DataFrame.from_dict(bad_budget_stats, orient="columns")
        bad_budget_df["claim"] = get_readable_fact_str(bad_fact, label_map)

        budget_df = pd.concat([good_budget_df, bad_budget_df], axis=0)

        budget_df.to_csv(f"{SAVE_FOLDER}/{exp_name}_budget_stats.csv", index=True)
