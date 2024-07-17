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
import os

label_map_path = "./Kelpie_package/Kelpie/data/FB15k-237/entity2wikidata.json"

label_map = json.load(open(label_map_path))

NUM_EPOCHS = 10
BASE_RES_FOLDER = "results/results/simulations/FB15k-237"

experiment_pairs = [
    ("/m/0151w_-/film/director/film-/m/0h03fhx","/m/0151w_-/film/director/film-/m/07kh6f3"), # 3_flow_1
    ("/m/06pj8-/film/director/film-/m/0260bz", "/m/06pj8-/film/director/film-/m/07024"),# 3_flow_2
    ("/m/014zcr-/film/actor/film./film/performance/film-/m/0418wg", "/m/014zcr-/film/actor/film./film/performance/film-/m/07w8fz"), #3_flow_3
    ("/m/0151w_-/film/actor/film./film/performance/film-/m/0pc62", "/m/0151w_-/film/actor/film./film/performance/film-/m/0h03fhx") #3_flow_4
]

clean_experiment_pairs = []
for pair in experiment_pairs:
    good_fact, bad_fact = pair
    good_fact = good_fact.split('-')
    bad_fact = bad_fact.split('-')
    clean_experiment_pairs.append((good_fact, bad_fact))

experiment_pairs = clean_experiment_pairs


strategies = ["approx_greedy", "multi_greedy", "random"]

baselines = ["random"]

tables = get_stats_tables(
    experiment_pairs=experiment_pairs,
    base_res_folder=BASE_RES_FOLDER,
    stats_file_name="stats.json",
    label_map=label_map,
    strategy_extension="random_approx_greedy",
)

all_exp_results = get_full_exp_results(
    experiment_pairs=experiment_pairs,
    strategies=strategies,
    n_epochs=NUM_EPOCHS,
    label_map=label_map,
    base_res_folder=BASE_RES_FOLDER,
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
    n_epochs=NUM_EPOCHS, 
    budget_folder="new_budgets", 
    label_map=label_map, 
    base_res_folder=BASE_RES_FOLDER
)

print("cost spent:", cost_spent)

all_rankings = simplify_full_exp_results(all_exp_results, NUM_EPOCHS)

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
    f"{plots_folder}fb15k_237_rankings_new.png",
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
    f"{plots_folder}fb15k_237_costs.png",
    label_map,
    {
        "approx_greedy": "Greedy",
        "random": "Random",
        "multi_greedy": "MultiObjective Greedy",
    },
)

