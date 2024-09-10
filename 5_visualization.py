import json
import os
import argparse
from helpers.plotting_utilities import (
    simplify_full_exp_results,
    get_stats_tables,
    plot_matrix_rankings,
    get_cost_spent_records,
    get_full_exp_results,
    get_disinformer_mitigator_avg_difference,
    plot_matrix_diffs,
    plot_costs_matrix,
)
from helpers.helpers import get_data_from_kg_name

def main(kg_name, experiment_pairs_file):
    dataset, TRAIN_PATH, TEST_PATH, VALID_PATH, LABEL_MAP_PATH = get_data_from_kg_name(kg_name)
    label_map = json.load(open(LABEL_MAP_PATH))

    num_epochs = 10
    base_res_folder = f"results/results/simulations/{kg_name}"

    with open(experiment_pairs_file, 'r') as f:
        experiment_pairs = json.load(f)


    strategies = ["approx_greedy", "multi_greedy", "random"]
    baselines = ["random"]

    tables = get_stats_tables(
        experiment_pairs=experiment_pairs,
        base_res_folder=base_res_folder,
        stats_file_name="stats.json",
        label_map=label_map,
        strategy_extension="random_approx_greedy",
    )

    all_exp_results = get_full_exp_results(
        experiment_pairs=experiment_pairs,
        strategies=strategies,
        n_epochs=num_epochs,
        label_map=label_map,
        base_res_folder=base_res_folder,
    )

    all_diffs = get_disinformer_mitigator_avg_difference(
        all_exp_results=all_exp_results, 
        experiment_pairs=experiment_pairs, 
        strategies=strategies, 
        stat_tables=tables, 
        label_map=label_map
    )

    cost_spent = get_cost_spent_records(
        experiment_pairs=experiment_pairs, 
        strategies=strategies, 
        n_epochs=num_epochs, 
        budget_folder="new_budgets", 
        label_map=label_map, 
        base_res_folder=base_res_folder
    )

    print("cost spent:", cost_spent)

    all_rankings = simplify_full_exp_results(all_exp_results, num_epochs)

    plots_folder = "results/plots/"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    plot_matrix_diffs(
        strategies,
        baselines,
        experiment_pairs,
        all_diffs,
        f"{plots_folder}fb15k_237_diffs_new_mogreedy_new.png",
        label_map,
        {
            "approx_greedy": "Greedy",
            "random": "Random",
            "multi_greedy": "MultiObjective Greedy",
        },
        with_cost=False,
        plot_stats_tables=False,
        plot_random_random=False,
        plot_topics_flipped=False,
        with_auc=False,
    )

    plot_matrix_rankings(
        strategies,
        baselines,
        experiment_pairs,
        all_rankings,
        f"{plots_folder}fb15k_237_rankings_new.png",
        label_map,
        with_cost=False,
        plot_stats_tables=False,
        plot_random_random=False,
    )

    plot_costs_matrix(
        cost_spent,
        experiment_pairs,
        strategies,
        f"{plots_folder}fb15k_237_costs.png",
        label_map,
        {
            "approx_greedy": "Greedy",
            "random": "Random",
            "multi_greedy": "MultiObjective Greedy",
        },
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process knowledge graph name and experiment pairs file.')
    parser.add_argument('--kg_name', type=str, help='The name of the knowledge graph')
    parser.add_argument('--experiment_pairs_file', type=str, help='The path to the JSON file containing experiment pairs')

    args = parser.parse_args()
    main(args.kg_name, args.experiment_pairs_file)
