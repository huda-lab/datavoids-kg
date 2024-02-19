import json
import pandas as pd
from Kelpie.dataset import Dataset
from helpers.plotting_utilities import get_base_exp_name
from helpers.helpers import get_readable_fact_str

# experiment_pairs = [
#     [
#         ('friedrich_hayek', 'iscitizenof', 'united_kingdom'),
#         ('friedrich_hayek', 'iscitizenof', 'austria')
#     ],
#     [
#         ('john_burridge', 'isaffiliatedto', 'manchester_city_f.c.'),
#         ('john_burridge', 'isaffiliatedto', 'sheffield_united_f.c.')
#     ],
#     # [
#     #     ('franz_kafka', 'iscitizenof', 'austria-hungary'),
#     #     ('franz_kafka', 'iscitizenof', 'czechoslovakia')
#     # ],
# ]
experiment_pairs = [
    [
        ('friedrich_hayek','iscitizenof', 'united_kingdom'),
        ('friedrich_hayek', 'iscitizenof', 'austria')
    ],
    [
        ('china', 'exports', 'wordnet_fabric_103309808'),
        ('china', 'exports', 'wordnet_apparel_102728440')
    ],
    # [
    #     ('john_burridge', 'isaffiliatedto', 'manchester_city_f.c.'),
    #     ('john_burridge', 'isaffiliatedto', 'sheffield_united_f.c.')
    # ]
]

DATASET_NAME = "yago310"
DATASET_PATH = 'Kelpie_package/Kelpie/data/YAGO3-10'
TRAIN_PATH = DATASET_PATH + '/train.txt'
TEST_PATH = DATASET_PATH + '/test.txt'
VALID_PATH = DATASET_PATH + '/valid.txt'

dataset = Dataset(
    name=DATASET_NAME,
    separator="\t",
    load=True,
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    valid_path=VALID_PATH,
)

BUDGET_FOLDER = "budgets"
NUM_ATTACK_BUDGET = 25


label_map = None

budget_df = pd.DataFrame(
    columns=["claim", "budget_degrees", "budget_relevance", "budget_costs"]
)
for good_fact, bad_fact in experiment_pairs:

    exp_name = get_base_exp_name(good_fact, bad_fact, label_map)
    with open(f"{BUDGET_FOLDER}/{exp_name}_budget.json", "r") as inf:
        budget = json.load(inf)
        good_budget_stats = {}
        for k, v in budget["good_budget_degrees"].items():
            print(get_readable_fact_str(k.split("-"), label_map))
        
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

        budget_df = pd.concat([budget_df, good_budget_df, bad_budget_df], axis=0)

budget_df.to_csv("yago310_budget_stats.csv", index=True)
