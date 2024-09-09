import json
import os
import pandas as pd
import argparse
from Kelpie.dataset import Dataset
from helpers.plotting_utilities import get_base_exp_name
from helpers.helpers import get_readable_fact_str,get_data_from_kg_name
from helpers.candidate_selection_helpers import tuple_to_filename, convert_relation_to_fn

def main(kg_name, experiment_pairs_file, save_folder="./results/stats", budget_folder="./results/generated_candidates"):
    dataset, TRAIN_PATH, TEST_PATH, VALID_PATH, LABEL_MAP_PATH = get_data_from_kg_name(kg_name)
    label_map = json.load(open(LABEL_MAP_PATH))

    with open(experiment_pairs_file, 'r') as f:
        experiment_pairs = json.load(f)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for x in experiment_pairs: 
        print(x)

    for good_fact, bad_fact in experiment_pairs:
        budget_df = pd.DataFrame(
            columns=["claim", "budget_degrees", "budget_relevance", "budget_costs"]
        )
        print("good fact",good_fact)
        relation = good_fact[1]
        exp_name = f"{tuple_to_filename(good_fact)}_{tuple_to_filename(bad_fact)}"
        budget_file_name = f"{budget_folder}/{kg_name}/{convert_relation_to_fn(relation)}/{exp_name}_budget.json"
        with open(budget_file_name, "r") as inf:
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

            budget_df.to_csv(f"{save_folder}/{exp_name}_budget_stats.csv", index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset path, dataset name, budget folder, save folder, experiment pairs file, and optional label map path.')
    parser.add_argument('--kg_name', type=str, required=True, help='The name of the knowledge graph')
    parser.add_argument('--experiment_pairs_file', type=str, required=True, help='The file containing experiment pairs')
    parser.add_argument('--save_folder', type=str, default="./results/stats", help='The folder where the statistics will be saved')
    parser.add_argument('--budget_folder', type=str, default="./results/generated_candidates", help='The folder where the budget files are located')

    args = parser.parse_args()
    main(args.kg_name, args.experiment_pairs_file, args.save_folder, args.budget_folder)

