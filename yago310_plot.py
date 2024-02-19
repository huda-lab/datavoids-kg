import json
from helpers.plotting_utilities import (
    simplify_full_exp_results,
    get_stats_tables,
    plot_matrix_rankings,
    get_cost_spent_records,
    get_full_exp_results,
    get_disinformer_mitigator_avg_difference,
    plot_matrix_diffs,
    get_ranking_aucs,
)
import traceback

label_map = None

NUM_EPOCHS = 5

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

strategies = ["multi_greedy", "approx_greedy", "random"]
baselines = ["random"]

# results_folder = "results/results/yago310/new_mo_strategy_2"
# results_folder = "results/results/yago310/new_mo_strategy_3"
results_folder = "results/results/yago310/new_mo_strategy_4"


try:
    tables = get_stats_tables(
        experiment_pairs=experiment_pairs, 
        label_map=label_map, 
        base_res_folder=results_folder, 
        stats_file_name="stats.json", 
        stats_field=None, 
        strategy_extension="approx_greedy_approx_greedy"
    )
except Exception as e:
    print("Error in get_stats_tables:", e)
    #traceback.print_exc()

try:
    all_exp_results = get_full_exp_results(
        experiment_pairs=experiment_pairs, 
        strategies=strategies, 
        n_epochs=NUM_EPOCHS,
        label_map=None,
        base_res_folder=results_folder,
        print_err=False
    )
except Exception as e:
    print("Error in get_full_exp_results:", e)
    #traceback.print_exc()

try:
    all_diffs = get_disinformer_mitigator_avg_difference(
        all_exp_results=all_exp_results, 
        experiment_pairs=experiment_pairs, 
        strategies=strategies, 
        label_map=None, 
        stat_tables=tables
    )
except Exception as e:
    print("Error in get_disinformer_mitigator_avg_difference:", e)
    #traceback.print_exc()

try:
    cost_spent = get_cost_spent_records(
        experiment_pairs=experiment_pairs, 
        label_map=None, 
        strategies=strategies, 
        n_epochs=NUM_EPOCHS, 
        base_res_folder=results_folder
    )
except Exception as e:
    print("Error in get_cost_spent_records:", e)
    #traceback.print_exc()

try:
    all_rankings = simplify_full_exp_results(
        full_exp_results=all_exp_results, 
        n_epochs=NUM_EPOCHS
    )
except Exception as e:
    print("Error in simplify_full_exp_results:", e)
    #traceback.print_exc()

disinformer_strategies = strategies
mitigator_strategies = list(set(strategies) - {"random"})

for disinformer_strategy in disinformer_strategies:
    try:
        ranking_aucs = get_ranking_aucs(
            experiment_pairs, strategies, baselines, all_rankings, label_map
        )
    except Exception as e:
        print(f"Error in get_ranking_aucs for {disinformer_strategy}:", e)
        #traceback.print_exc()

    try:
        plot_matrix_rankings(
            strategies=[disinformer_strategy],
            baseline_strategies=mitigator_strategies,
            experiment_pairs=experiment_pairs,
            simplified_exp_results=all_rankings,
            save_file=f"plots/new_facts_rankings_{disinformer_strategy}.png",
            label_map=label_map,
            stats_tables=tables,
            with_cost=True,
            cost_spent=cost_spent,
            plot_stats_tables=False,
            plot_random_random=False
        )
    except Exception as e:
        print(f"Error in plot_matrix_rankings for {disinformer_strategy}:", e)
        #traceback.print_exc()

try:
    plot_matrix_diffs(
        strategies=strategies,
        baseline_strategies=baselines,
        experiment_pairs=experiment_pairs,
        exp_diffs=all_diffs,
        save_file="plots/final_paper_plots/yago310_diffs_4.png",
        label_map=None,
        strategy_labels={
            "approx_greedy": "Greedy",
            "random": "Random",
            "multi_greedy": "MultiObjective Greedy",
        },
        with_cost=False,
        plot_stats_tables=False,
        plot_random_random=False,
        plot_topics_flipped=False,
        with_auc=False
    )
except Exception as e:
    print("Error in plot_matrix_diffs:", e)
    #traceback.print_exc()
