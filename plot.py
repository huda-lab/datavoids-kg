import json
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

label_map_path = "entity2wikidata.json"
label_map = json.load(open(label_map_path))

NUM_EPOCHS = 10
BASE_RES_FOLDER = "results/final_paper_new_mo_greedy"

exp_inputs = [1, 2, 6, 7, 10]
experiment_pairs = []
for i in exp_inputs:
    with open(f"experiment_inputs/input_{i}.txt", "r", encoding="utf-8") as f:
        good_fact, bad_fact = f.readlines()
        good_fact = good_fact.strip().split("-")
        bad_fact = bad_fact.strip().split("-")
        experiment_pairs.append((good_fact, bad_fact))

strategies = ["approx_greedy", "multi_greedy", "random"]

baselines = ["random"]

tables = get_stats_tables(
    experiment_pairs,
    BASE_RES_FOLDER,
    "stats.json",
    label_map=label_map,
    strategy_extension="random_approx_greedy",
)

all_exp_results = get_full_exp_results(
    experiment_pairs,
    strategies,
    NUM_EPOCHS,
    label_map,
    BASE_RES_FOLDER,
)

all_diffs = get_disinformer_mitigator_avg_difference(
    all_exp_results, experiment_pairs, strategies, tables, label_map
)

cost_spent = get_cost_spent_records(
    experiment_pairs, strategies, NUM_EPOCHS, "new_budgets", label_map, BASE_RES_FOLDER
)

print("cost spent:", cost_spent)

all_rankings = simplify_full_exp_results(all_exp_results, NUM_EPOCHS)

plot_matrix_diffs(
    strategies,
    baselines,
    experiment_pairs,
    all_diffs,
    "plots/final_paper_plots/fb15k_237_diffs_new_mogreedy_new.png",
    label_map,
    {
        "approx_greedy": "Greedy",
        "random": "Random",
        "multi_greedy": "MultiObjective Greedy",
    },
    # tables,
    with_cost=False,
    # cost_spent=cost_spent,
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
    "plots/final_paper_plots/fb15k_237_rankings_new.png",
    label_map,
    # {
    #     "approx_greedy": "Greedy",
    #     "random": "Random",
    #     "multi_greedy": "MultiObjective Greedy",
    # },
    # tables,
    with_cost=False,
    # cost_spent=cost_spent,
    plot_stats_tables=False,
    plot_random_random=False,
    # plot_topics_flipped=False,
    # with_auc=False,
)

plot_costs_matrix(
    cost_spent,
    experiment_pairs,
    strategies,
    "plots/final_paper_plots/fb15k_237_costs.png",
    label_map,
    {
        "approx_greedy": "Greedy",
        "random": "Random",
        "multi_greedy": "MultiObjective Greedy",
    },
)
