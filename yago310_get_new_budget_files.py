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
train_test_valid_paths = [TRAIN_PATH, TEST_PATH, VALID_PATH]

tuples_to_experiment_with = [
    # [
    #     ('friedrich_hayek', 'iscitizenof', 'united_kingdom'),
    #     ('friedrich_hayek', 'iscitizenof', 'austria')
    # ],
    # [
    #     ('john_burridge', 'isaffiliatedto', 'manchester_city_f.c.'),
    #     ('john_burridge', 'isaffiliatedto', 'sheffield_united_f.c.')
    # ],
    [
        ('franz_kafka', 'iscitizenof', 'austria-hungary'),
        ('franz_kafka', 'iscitizenof', 'czechoslovakia')
    ],
]


for tuple_values in tuples_to_experiment_with:

    good_fact, bad_fact = tuple_values


    EXPERIMENT_NAME = f"{good_fact}_{bad_fact}"
    BUDGET_FILE = f"budgets/{EXPERIMENT_NAME}_budget_with_relevance.json"
    NUM_ATTACK_BUDGET = 25

    reduced_dataset_path = f"reduced_datasets/yago310/{EXPERIMENT_NAME}.txt"

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
        f"results/yago310/{EXPERIMENT_NAME}_budget_model.pt",
        BUDGET_FILE,
        "kelpie",
        "degree",
    )