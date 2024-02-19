import json
from collections import defaultdict
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from helpers.helpers import calculate_auc
from helpers.constants import STATS_COLS

# plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 25
plt.rcParams["figure.dpi"] = 200
plt.rcParams["image.cmap"] = "tab10"
TABLEAU_PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=TABLEAU_PALETTE)


def get_base_exp_name(good_fact, bad_fact, label_map=None):
    if not label_map:
        return f"{good_fact}_{bad_fact}"

    return f"{label_map[good_fact[0]]['label']}_{label_map[good_fact[2]]['label']}_{label_map[bad_fact[2]]['label']}"


def get_shortened_name(good_fact: str, bad_fact: str, label_map: dict):
    if label_map:
        mitigator_entity = label_map[good_fact[2]]["label"]
        mitigator_short = " ".join(mitigator_entity.split(" ")[:2])
        disinformer_entity = label_map[bad_fact[2]]["label"]
        disinformer_short = " ".join(disinformer_entity.split(" ")[:2])
        return f"{mitigator_short} vs {disinformer_short}"
    return f"{good_fact[2]} vs {bad_fact[2]}"


def plot_facts_added_difference(facts_added_1, facts_added_2):
    to_plot = [facts_added_1[i] == facts_added_2[i] for i in range(len(facts_added_1))]
    plt.plot(to_plot, color="black")
    plt.title("Difference between facts added")


def get_stats_tables(
    experiment_pairs: list,
    base_res_folder: str,
    stats_file_name: str = "stats.json",
    stats_field=None,
    label_map: dict = None,
    strategy_extension: str = "random_greedy",
):
    tables = []
    for good_fact, bad_fact in experiment_pairs:
        base_experiment_name = get_base_exp_name(good_fact, bad_fact, label_map)
        try:
            res_folder = (
                f"{base_res_folder}/{base_experiment_name}_{strategy_extension}"
            )
            with open(f"{res_folder}/{stats_file_name}", "r") as infile:
                data = json.load(infile)
                if stats_field is not None:
                    data = data[stats_field]
                data["good_explanations_degree_head"] = data[
                    "good_explanations_degree"
                ]["head"]
                data["bad_explanations_degree_head"] = data["bad_explanations_degree"][
                    "head"
                ]
                data["good_explanations_degree_tail"] = data[
                    "good_explanations_degree"
                ]["tail"]
                data["bad_explanations_degree_tail"] = data["bad_explanations_degree"][
                    "tail"
                ]
                del data["good_explanations_degree"]
                del data["bad_explanations_degree"]

                for key, val in data.items():
                    data[key] = round(val, 2)
                tables.append(pd.DataFrame.from_records([data]))
        except:
            print("Could not get stats for", base_experiment_name)
            traceback.print_exc()
            continue
    return tables


def combine_all_stats_tables(stats_tables: list, experiment_pairs: list):
    res = pd.DataFrame(columns=STATS_COLS)
    for i, table in enumerate(stats_tables):
        good_fact, bad_fact = experiment_pairs[i]
        good_fact_df = pd.DataFrame(
            [
                [
                    " ".join(good_fact),
                    table.at[0, "good_fact_num_budget"].item(),
                    table.at[0, "good_entity_degree"].item(),
                    table.at[0, "head_degree"].item(),
                    table.at[0, "good_explanations_degree_head"].item(),
                    table.at[0, "good_explanations_degree_tail"].item(),
                ]
            ],
            columns=STATS_COLS,
        )
        bad_fact_df = pd.DataFrame(
            [
                [
                    " ".join(bad_fact),
                    table.at[0, "bad_fact_num_budget"].item(),
                    table.at[0, "bad_entity_degree"].item(),
                    table.at[0, "head_degree"].item(),
                    table.at[0, "bad_explanations_degree_head"].item(),
                    table.at[0, "bad_explanations_degree_tail"].item(),
                ]
            ],
            columns=STATS_COLS,
        )
        res = pd.concat([res, good_fact_df, bad_fact_df], ignore_index=True, axis=0)

    return res


def get_simplified_exp_results(
    strategies,
    experiment_pairs,
    label_map,
    base_res_folder="results/remove_overlapping_budget",
):
    exp_records = defaultdict(dict)
    for good_fact, bad_fact in experiment_pairs:
        base_experiment_name = get_base_exp_name(good_fact, bad_fact, label_map)
        for strategy_one in strategies:
            for strategy_two in strategies:
                try:
                    exp_type = f"{strategy_one}_{strategy_two}"
                    res_folder = f"{base_res_folder}/{base_experiment_name}_{exp_type}"
                    with open(
                        f"{res_folder}/results_auc_rankings.json", "r", encoding="utf-8"
                    ) as infile:
                        data = json.load(infile)
                        exp_records[base_experiment_name][exp_type] = [
                            data["good_entity_rankings"],
                            data["bad_entity_rankings"],
                            data["AUC_mitigator"],
                            data["AUC_disinformer"],
                        ]
                except:
                    print(
                        "Failed to retrieve simplified experiment records for",
                        res_folder,
                    )
                    continue
        try:
            res_folder = f"{base_res_folder}/{base_experiment_name}_random_random_1"
            with open(
                f"{res_folder}/results_auc_rankings.json", "r", encoding="utf-8"
            ) as infile:
                data = json.load(infile)
                exp_records[base_experiment_name]["random_random_1"] = [
                    data["good_entity_rankings"],
                    data["bad_entity_rankings"],
                    data["AUC_mitigator"],
                    data["AUC_disinformer"],
                ]
        except:
            print("Cannot retrieve random-random 1 results for ", base_experiment_name)
            continue

        try:
            res_folder = f"{base_res_folder}/{base_experiment_name}_random_random_2"
            with open(
                f"{res_folder}/results_auc_rankings.json", "r", encoding="utf-8"
            ) as infile:
                data = json.load(infile)
                exp_records[base_experiment_name]["random_random_2"] = [
                    data["good_entity_rankings"],
                    data["bad_entity_rankings"],
                    data["AUC_mitigator"],
                    data["AUC_disinformer"],
                ]
        except:
            print("Cannot retrieve random-random 2 results for ", base_experiment_name)
            continue

    return exp_records

def get_experiment_type(strategy_one, strategy_two, attempt_flip=False):
    """
    Generate the experiment type identifier based on the strategy order.
    If the first attempt fails (indicated by attempt_flip=True), flip the order.
    """
    if attempt_flip:
        # If flipping is needed, swap the strategies
        strategy_one, strategy_two = strategy_two, strategy_one
    return f"{strategy_one}_{strategy_two}"


def get_fact_cost_from_budget(fact_added: tuple, budget: dict, good_or_bad: str):
    return budget[f"{good_or_bad}_budget_costs"]["-".join(fact_added)]


def get_cost_spent_records(
    experiment_pairs: list,
    strategies: list,
    n_epochs: int,
    budget_folder: str,
    label_map: dict = None,
    base_res_folder: str = "results/remove_overlapping_budget",
):
    cost_spent = defaultdict(dict)

    for good_fact, bad_fact in experiment_pairs:
        base_experiment_name = get_base_exp_name(good_fact, bad_fact, label_map)
        budget_file_name = f"{budget_folder}/{base_experiment_name}_budget.json"
        with open(budget_file_name, "r") as f:
            budget = json.load(f)

        if not budget:
            raise Exception("Budget could not be loaded")
        for strategy_one in strategies:
            for strategy_two in strategies:
                exp_type = f"{strategy_one}_{strategy_two}"
                print("exp type:", exp_type)
                res_folder = f"{base_res_folder}/{base_experiment_name}_{exp_type}"
                try:
                    with open(
                        f"{res_folder}/all_results.json", "r", encoding="utf-8"
                    ) as infile:
                        data = json.load(infile)
                        attack_budget = len(data[0]) - 2
                        print("attack budget:", attack_budget)
                        cost_spent[base_experiment_name][exp_type] = {
                            "mitigator_cost": [0] * (attack_budget + 1),
                            "disinformer_cost": [0] * (attack_budget + 1),
                        }
                        good_costs = []
                        bad_costs = []
                        if strategy_one == "random" or strategy_two == "random":
                            num_epochs = n_epochs
                        else:
                            num_epochs = 1
                        for e in range(num_epochs):
                            good_costs.append([])
                            bad_costs.append([])
                            epoch = data[e]
                            # if "0" not in e:
                            #     e["0"] = data[0]["0"]
                            for r in range(attack_budget + 1):
                                if r == 0:
                                    good_costs[e].append(0)
                                    bad_costs[e].append(0)
                                else:
                                    # accumulate cost spent from previous round
                                    good_costs[e].append(
                                        get_fact_cost_from_budget(
                                            epoch[str(r)]["good_fact_added"],
                                            budget,
                                            "good",
                                        )
                                        + good_costs[e][r - 1]
                                    )
                                    # print(
                                    #     "adding: ",
                                    #     get_fact_cost_from_budget(
                                    #         epoch[str(r)]["good_fact_added"],
                                    #         budget,
                                    #         "good",
                                    #     ),
                                    # )
                                    # print(
                                    #     "new res: ",
                                    #     get_fact_cost_from_budget(
                                    #         epoch[str(r)]["good_fact_added"],
                                    #         budget,
                                    #         "good",
                                    #     )
                                    #     + good_costs[e][r - 1],
                                    # )
                                    # print(
                                    #     "current spent cost:",
                                    #     e[str(r)]["mitigator_spent_cost"],
                                    # )
                                    bad_costs[e].append(
                                        get_fact_cost_from_budget(
                                            epoch[str(r)]["bad_fact_added"],
                                            budget,
                                            "bad",
                                        )
                                        + bad_costs[e][r - 1]
                                    )

                                # prev_mitigator_cost = (
                                #     e[str(r - 1)]["mitigator_spent_cost"]
                                #     if r > 0
                                #     else 0
                                # )
                                # prev_disinformer_cost = (
                                #     e[str(r - 1)]["disinformer_spent_cost"]
                                #     if r > 0
                                #     else 0
                                # )
                                # cost_spent[base_experiment_name][exp_type][
                                #     "mitigator_cost"
                                # ][r] = (
                                #     e[str(r)]["mitigator_spent_cost"]
                                #     + prev_mitigator_cost
                                # )
                                # cost_spent[base_experiment_name][exp_type][
                                #     "disinformer_cost"
                                # ][r] = (
                                #     e[str(r)]["disinformer_spent_cost"]
                                #     + prev_disinformer_cost
                                # )
                        cost_spent[base_experiment_name][exp_type][
                            "mitigator_cost"
                        ] = np.average(
                            good_costs,
                            axis=0,
                        )

                        cost_spent[base_experiment_name][exp_type][
                            "disinformer_cost"
                        ] = np.average(
                            bad_costs,
                            axis=0,
                        )
                except:
                    print("Cannot retrieve cost for", base_experiment_name, exp_type)
                    traceback.print_exc()
                    continue
    return cost_spent


def plot_costs_matrix(
    cost_spent: dict,
    experiment_pairs: list,
    strategies: list,
    save_file: str,
    label_map: dict = None,
    strategy_labels: dict = None,
):
    if not strategy_labels:
        strategy_labels = {x: x for x in strategies}
    fig, ax = plt.subplots(
        1,
        len(experiment_pairs),
        figsize=(len(experiment_pairs) * 13, 10),
        sharey=False,
        sharex=False,
    )
    j = 0
    for good_fact, bad_fact in experiment_pairs:
        base_exp_name = get_base_exp_name(good_fact, bad_fact, label_map)
        k = 0
        for strategy in strategies:
            try:
                shortened_name = get_shortened_name(good_fact, bad_fact, label_map)
                ax[j].set_yscale("log")
                ax[j].plot(
                    cost_spent[base_exp_name][f"{strategy}_random"]["mitigator_cost"],
                    color=TABLEAU_PALETTE[k],
                    label=strategy_labels[strategy],
                )
                ax[j].set_title(shortened_name)
            except Exception as e:
                print(e)
            k += 1
        ax[j].legend(loc="lower right")
        j += 1
    fig.savefig(save_file)


def simplify_full_exp_results(full_exp_results: dict, n_epochs: int):
    simplified_exp_results = defaultdict(dict)
    for base_exp_name in full_exp_results:
        current_exps = full_exp_results[base_exp_name]
        simplified_exp_results[base_exp_name] = {}
        for exp_type in current_exps:
            try:
                good_rankings = []
                bad_rankings = []
                attack_budget = len(current_exps[exp_type][0]) - 1
                if "random" in exp_type:
                    num_epochs = n_epochs
                else:
                    num_epochs = 1
                for e in range(num_epochs):
                    good_rankings.append([])
                    bad_rankings.append([])
                    for r in range(attack_budget):
                        good_rankings[e].append(
                            1 / current_exps[exp_type][e][str(r)]["good_entity_rank"]
                        )
                        bad_rankings[e].append(
                            1 / current_exps[exp_type][e][str(r)]["bad_entity_rank"]
                        )
                simplified_exp_results[base_exp_name][exp_type] = [
                    np.average(good_rankings, axis=0),
                    np.average(bad_rankings, axis=0),
                ]
            except:
                print("failed to simplify full results at", base_exp_name, exp_type)
                print()
                continue
    return simplified_exp_results

def impute_missing_first_round(exp_results: list):
    for i in range(len(exp_results)):
        if "0" not in exp_results[i]:
            exp_results[i]["0"] = exp_results[0]["0"]
    return exp_results


def get_full_exp_results(
    experiment_pairs: list,
    strategies: list,
    n_epochs: int,
    label_map: dict = None,
    base_res_folder: str = "results/remove_overlapping_budget",
    print_err: bool = False,
):
    exp_records = defaultdict(dict)
    for good_fact, bad_fact in experiment_pairs:
        base_experiment_name = get_base_exp_name(good_fact, bad_fact, label_map)
        for strategy_one in strategies:
            for strategy_two in strategies:
                try:
                    exp_type = f"{strategy_one}_{strategy_two}"
                    res_folder = f"{base_res_folder}/{base_experiment_name}_{exp_type}"
                    with open(
                        f"{res_folder}/all_results.json", "r", encoding="utf-8"
                    ) as infile:
                        data = json.load(infile)
                        if strategy_one == "random" or strategy_two == "random":
                            num_epochs = n_epochs
                        else:
                            num_epochs = 1
                        exp_records[base_experiment_name][
                            exp_type
                        ] = impute_missing_first_round(data[:num_epochs])
                except:
                    if print_err:
                        print(
                            "Failed to retrieve full experiment records for", res_folder
                        )
                    continue
        try:
            res_folder_1 = f"{base_res_folder}/{base_experiment_name}_random_random_1"
            with open(f"{res_folder_1}/all_results.json", "r") as infile:
                data = json.load(infile)
                exp_records[base_experiment_name]["random_random_1"] = data[:n_epochs]
            res_folder_2 = f"{base_res_folder}/{base_experiment_name}_random_random_2"
            with open(f"{res_folder_2}/all_results.json", "r") as infile:
                data = json.load(infile)
                exp_records[base_experiment_name]["random_random_2"] = data[:n_epochs]
        except:
            if print_err:
                print(
                    "Failed to retrieve full experiment records for",
                    base_experiment_name,
                    "random-random 1 or 2",
                )
            continue
    return exp_records


def get_avg_disinformer_mitigator_difference_1_exp(exp_results, num_rounds):
    all_diffs = []

    for epoch in exp_results:
        epoch_diffs = []
        for round_num in range(int(num_rounds)):
            round_ = epoch[str(int(round_num))]
            # bad entity rank - good entity rank
            # +ve if bad entity rank > good entity rank -> good entity is doing better
            # -ve if bad < good -> bad entity is doing better
            epoch_diffs.append(
                (1 / round_["good_entity_rank"]) - (1 / round_["bad_entity_rank"])
            )
        all_diffs.append(epoch_diffs)

    return np.average(np.array(all_diffs), axis=0)


def get_disinformer_mitigator_avg_difference(
    all_exp_results,
    experiment_pairs,
    strategies,
    stat_tables,
    label_map=None,
    print_err: bool = False,
):
    all_diffs = defaultdict(dict)
    j = 0
    for good_fact, bad_fact in experiment_pairs:
        base_experiment_name = get_base_exp_name(good_fact, bad_fact, label_map)
        # add one for round 0
        num_rounds = stat_tables[j].at[0, "good_fact_num_budget"] + 1
        j += 1
        for strategy_one in strategies:
            for strategy_two in strategies:
                try:
                    exp_type = f"{strategy_one}_{strategy_two}"
                    all_diffs[base_experiment_name][
                        exp_type
                    ] = get_avg_disinformer_mitigator_difference_1_exp(
                        all_exp_results[base_experiment_name][exp_type], num_rounds
                    )
                except:
                    if print_err:
                        print(
                            "Failed to get disinformer-mitigator avg difference at",
                            base_experiment_name,
                            strategy_one,
                            "_",
                            strategy_two,
                        )
                        # traceback.print_exc()
                        print()
        try:
            all_diffs[base_experiment_name][
                "random_random_1"
            ] = get_avg_disinformer_mitigator_difference_1_exp(
                all_exp_results[base_experiment_name]["random_random_1"], num_rounds
            )

            all_diffs[base_experiment_name][
                "random_random_2"
            ] = get_avg_disinformer_mitigator_difference_1_exp(
                all_exp_results[base_experiment_name]["random_random_2"], num_rounds
            )
        except:
            if print_err:
                print(
                    "Failed to get disinformer-mitigator avg difference at",
                    base_experiment_name,
                    "random_random_1 or 2",
                )
            continue

    return all_diffs


def plot_rankings(
    ax: Axes, random_rankings: list, exp_rankings: list, mitigator_strategy: str
):
    ax.plot(random_rankings, color="blue", label="Random")
    ax.plot(exp_rankings, color="red", label=mitigator_strategy)
    ax.fill_between(
        [a for a in range(len(random_rankings))],
        random_rankings,
        exp_rankings,
        where=(np.array(random_rankings) - np.array(exp_rankings)) <= 0,
        color="green",
        alpha=0.2,
    )

    ax.fill_between(
        [a for a in range(len(exp_rankings))],
        exp_rankings,
        random_rankings,
        where=(np.array(random_rankings) - np.array(exp_rankings)) > 0,
        color="red",
        alpha=0.2,
    )
    ax.legend(loc="lower right")


def run_plot_stats_table(
    ax: Axes,
    stats_table: pd.DataFrame,
    colWidth: float = 0.5,
    loc: str = "right",
    fontsize: int = 19,
):
    ax.table(
        stats_table.to_numpy(),
        loc=loc,
        colWidths=list([colWidth] * len(stats_table.columns)),
        colLabels=stats_table.columns,
        rowLabels=stats_table.index,
        fontsize=fontsize,
    )
    ax.axis("tight")
    ax.axis("off")


def plot_matrix_rankings(
    strategies: list,
    baseline_strategies: list,
    experiment_pairs: list,
    simplified_exp_results: dict,
    save_file: str,
    label_map: dict = None,
    stats_tables: list = None,
    with_cost: bool = False,
    cost_spent: dict = None,
    plot_stats_tables=False,
    plot_random_random: bool = False,
):
    strategies_no_baseline = list(set(strategies) - set(baseline_strategies))
    # num additional rows for stats table and random-random plot
    n_additional_rows = 0
    if plot_stats_tables:
        n_additional_rows += 1
    if plot_random_random:
        n_additional_rows += 1

    nrows = (len(strategies) * len(strategies_no_baseline)) + n_additional_rows
    ncols = len(experiment_pairs) * 2

    fig, ax = plt.subplots(
        nrows, ncols, figsize=(ncols * 13, nrows * 10), sharey=False, sharex=False
    )

    ax[1][0].set_ylabel("Random vs Random")
    for j, _ in enumerate(experiment_pairs):
        try:
            current_row = 0
            good_fact, bad_fact = experiment_pairs[j]
            ax[0][j * 2].set_title(get_shortened_name(good_fact, bad_fact, label_map))
            # pylint: disable=arguments-out-of-order
            ax[0][j * 2 + 1].set_title(
                get_shortened_name(bad_fact, good_fact, label_map)
            )
            if plot_stats_tables:
                run_plot_stats_table(ax[current_row][j * 2], stats_tables[j])
                ax[current_row][j * 2 + 1].axis("off")
                current_row += 1
            base_experiment_name = get_base_exp_name(good_fact, bad_fact, label_map)
            # plotting random-random mitigator
            if plot_random_random:
                ax[current_row][j * 2].plot(
                    simplified_exp_results[base_experiment_name]["random_random_1"][0],
                    color="blue",
                    label="Random 1",
                )
                ax[current_row][j * 2].plot(
                    simplified_exp_results[base_experiment_name]["random_random_2"][0],
                    color="red",
                    label="Random 2",
                )
                ax[current_row][j * 2].legend(loc="lower right")
        except:
            print("Failed to plot", base_experiment_name, "random-random plotting")
            continue

    # for every disinformer strategy
    for i, _ in enumerate(strategies):
        disinformer_strategy = strategies[i]
        # for every mitigator strategy
        for m, _ in enumerate(strategies_no_baseline):
            mitigator_strategy = strategies_no_baseline[m]
            row = ax[i * (len(strategies_no_baseline)) + m + n_additional_rows]

            row[0].set_ylabel(
                f"Disinformer {disinformer_strategy}\n Mitigator {mitigator_strategy}"
            )
            for j in range(0, ncols, 2):
                try:
                    good_fact, bad_fact = experiment_pairs[j // 2]

                    base_experiment_name = get_base_exp_name(
                        good_fact, bad_fact, label_map
                    )
                    exp_rankings = simplified_exp_results[base_experiment_name][
                        f"{mitigator_strategy}_{disinformer_strategy}"
                    ][0]
                    random_rankings = simplified_exp_results[base_experiment_name][
                        f"random_{disinformer_strategy}"
                    ][0]
                    plot_rankings(
                        row[j],
                        random_rankings,
                        exp_rankings,
                        mitigator_strategy,
                    )
                    # plot_auc_diff(row[j], exp_rankings, random_rankings)
                    if with_cost and cost_spent is not None:
                        cost_exp = cost_spent[base_experiment_name][
                            f"{mitigator_strategy}_{disinformer_strategy}"
                        ]["mitigator_cost"]
                        cost_random = cost_spent[base_experiment_name][
                            f"random_{disinformer_strategy}"
                        ]["mitigator_cost"]
                        plot_cost(row[j], cost_exp, cost_random, mitigator_strategy)

                    # topic flipped
                    exp_rankings = simplified_exp_results[base_experiment_name][
                        f"{disinformer_strategy}_{mitigator_strategy}"
                    ][1]
                    random_rankings = simplified_exp_results[base_experiment_name][
                        f"{disinformer_strategy}_random"
                    ][1]
                    plot_rankings(
                        row[j + 1], random_rankings, exp_rankings, mitigator_strategy
                    )
                    # plot_auc_diff(row[j+1], exp_rankings, random_rankings)
                    if with_cost and cost_spent is not None:
                        cost_exp = cost_spent[base_experiment_name][
                            f"{disinformer_strategy}_{mitigator_strategy}"
                        ]["disinformer_cost"]
                        cost_random = cost_spent[base_experiment_name][
                            f"{disinformer_strategy}_random"
                        ]["disinformer_cost"]
                        plot_cost(row[j + 1], cost_exp, cost_random, mitigator_strategy)

                except:
                    print("Failed to plot matrix rankings")
                    # traceback.print_exc()
                    print()
                    continue

    fig.savefig(save_file)


def plot_matrix_diffs(
    strategies: list,
    baseline_strategies: list,
    experiment_pairs: list,
    exp_diffs: dict,
    save_file: str,
    label_map: dict = None,
    strategy_labels: dict = None,
    stats_tables: list = None,
    with_cost: bool = False,
    cost_spent: dict = None,
    plot_stats_tables: bool = False,
    plot_random_random: bool = False,
    plot_topics_flipped: bool = False,
    with_auc: bool = False,
):
    strategies_no_baseline = list(set(strategies) - set(baseline_strategies))

    if strategy_labels is None:
        strategy_labels = {s: s for s in strategies}
    # num additional rows for stats table and random-random plot
    n_additional_rows = 0
    if plot_stats_tables:
        n_additional_rows += 1
    if plot_random_random:
        n_additional_rows += 1

    nrows = (len(strategies) * len(strategies_no_baseline)) + n_additional_rows
    ncols = len(experiment_pairs)
    if plot_topics_flipped:
        ncols *= 2

    fig, ax = plt.subplots(
        nrows, ncols, figsize=(ncols * 13, nrows * 10), sharey=False, sharex=False
    )
    plt.subplots_adjust(wspace=0.5)
    ax[1][0].set_ylabel("Random vs Random")
    for j, _ in enumerate(experiment_pairs):
        try:
            current_row = 0
            good_fact, bad_fact = experiment_pairs[j]
            current_col = j
            if plot_topics_flipped:
                current_col = j * 2

            ax[current_row][current_col].set_title(
                get_shortened_name(good_fact, bad_fact, label_map)
            )
            if plot_topics_flipped:
                # pylint: disable=arguments-out-of-order
                ax[current_row][current_col + 1].set_title(
                    get_shortened_name(bad_fact, good_fact, label_map)
                )
            if plot_stats_tables:
                run_plot_stats_table(ax[current_row][current_col], stats_tables[j])
                if plot_topics_flipped:
                    ax[current_row][current_col + 1].axis("off")
                current_row += 1
            if plot_random_random:
                base_experiment_name = get_base_exp_name(good_fact, bad_fact, label_map)
                plot_random_random_diffs(
                    ax[current_row][current_col], base_experiment_name, exp_diffs
                )
        except Exception as e:
            print("Failed at plotting", e)
            continue

    # for every disinformer strategy
    for i, _ in enumerate(strategies):
        disinformer_strategy = strategies[i]
        # for every mitigator strategy
        for m, _ in enumerate(strategies_no_baseline):
            mitigator_strategy = strategies_no_baseline[m]
            row = ax[i * (len(strategies_no_baseline)) + m + n_additional_rows]

            row[0].set_ylabel(
                f"Disinformer {strategy_labels[disinformer_strategy]}\n Mitigator {strategy_labels[mitigator_strategy]}"
            )
            for k, _ in enumerate(experiment_pairs):
                try:
                    good_fact, bad_fact = experiment_pairs[k]
                    current_col = k
                    if plot_topics_flipped:
                        current_col *= 2

                    base_experiment_name = get_base_exp_name(
                        good_fact, bad_fact, label_map
                    )
                    # disinformer - mitigator
                    # if mitigator is above line = 0, then it's doing better than disinformer
                    exp_rankings = exp_diffs[base_experiment_name][
                        f"{mitigator_strategy}_{disinformer_strategy}"
                    ]
                    random_rankings = exp_diffs[base_experiment_name][
                        f"random_{disinformer_strategy}"
                    ]
                    plot_diffs(
                        exp_rankings,
                        random_rankings,
                        mitigator_strategy,
                        strategy_labels[mitigator_strategy],
                        row[current_col],
                    )
                    if with_auc:
                        plot_auc_diff(row[current_col], exp_rankings, random_rankings)
                    if with_cost and cost_spent is not None:
                        cost_exp = cost_spent[base_experiment_name][
                            f"{mitigator_strategy}_{disinformer_strategy}"
                        ]["mitigator_cost"]
                        cost_random = cost_spent[base_experiment_name][
                            f"random_{disinformer_strategy}"
                        ]["mitigator_cost"]
                        plot_cost(
                            row[current_col], cost_exp, cost_random, mitigator_strategy
                        )

                    # topic flipped
                    if plot_topics_flipped:
                        exp_rankings = (
                            np.array(
                                exp_diffs[base_experiment_name][
                                    f"{disinformer_strategy}_{mitigator_strategy}"
                                ]
                            )
                            * -1
                        )
                        random_rankings = (
                            np.array(
                                exp_diffs[base_experiment_name][
                                    f"{disinformer_strategy}_random"
                                ]
                            )
                            * -1
                        )
                        plot_diffs(
                            exp_rankings,
                            random_rankings,
                            mitigator_strategy,
                            strategy_labels[mitigator_strategy],
                            row[current_col + 1],
                        )
                        if with_auc:
                            plot_auc_diff(
                                row[current_col + 1], exp_rankings, random_rankings
                            )
                        if with_cost and cost_spent is not None:
                            cost_exp = cost_spent[base_experiment_name][
                                f"{disinformer_strategy}_{mitigator_strategy}"
                            ]["disinformer_cost"]
                            cost_random = cost_spent[base_experiment_name][
                                f"{disinformer_strategy}_random"
                            ]["disinformer_cost"]
                            plot_cost(
                                row[current_col + 1],
                                cost_exp,
                                cost_random,
                                mitigator_strategy,
                            )

                except:
                    # traceback.print_exc()
                    continue

    fig.savefig(save_file)


def plot_diffs(
    exp_rankings,
    random_rankings,
    mitigator_strategy: str,
    mitigator_strategy_label: str,
    ax,
):
    # plot 0 line
    ax.axhline(y=0, color="black", linestyle="-")
    ax.plot(random_rankings, color="blue", label="Random")
    ax.plot(exp_rankings, color="red", label=mitigator_strategy_label)
    ax.fill_between(
        [a for a in range(len(random_rankings))],
        random_rankings,
        exp_rankings,
        where=(np.array(exp_rankings) - np.array(random_rankings)) >= 0,
        color="green",
        alpha=0.2,
    )

    ax.fill_between(
        [a for a in range(len(exp_rankings))],
        exp_rankings,
        random_rankings,
        where=(np.array(exp_rankings) - np.array(random_rankings)) < 0,
        color="red",
        alpha=0.2,
    )
    ax.legend(loc="lower right")


def plot_cost(ax: Axes, cost_exp, cost_random, mitigator_strategy):
    cost_ax = ax.twinx()
    cost_ax.plot(cost_exp, color="red", alpha=0.5, label=mitigator_strategy)
    cost_ax.plot(cost_random, color="blue", alpha=0.5, label="random")


def get_auc_diff_1_exp(exp_rankings, random_rankings):
    exp_auc = calculate_auc(exp_rankings)
    random_auc = calculate_auc(random_rankings)
    return round(exp_auc - random_auc, 2)


def get_ranking_aucs(
    experiment_pairs,
    strategies,
    baseline,
    exp_results_simplified,
    label_map,
    print_err: bool = False,
) -> pd.DataFrame:
    res = []
    strategies_no_baseline = list(set(strategies) - set(baseline))
    for good_fact, bad_fact in experiment_pairs:
        base_experiment_name = get_base_exp_name(good_fact, bad_fact, label_map)
        for i, _ in enumerate(strategies):
            disinformer_strategy = strategies[i]
            for m, _ in enumerate(strategies_no_baseline):
                try:
                    mitigator_strategy = strategies_no_baseline[m]
                    exp_rankings = exp_results_simplified[base_experiment_name][
                        f"{mitigator_strategy}_{disinformer_strategy}"
                    ][0]
                    random_rankings = exp_results_simplified[base_experiment_name][
                        f"random_{disinformer_strategy}"
                    ][0]
                    res.append(
                        {
                            "mitigator_fact": " ".join(good_fact),
                            "disinformer_fact": " ".join(bad_fact),
                            "mitigator_strategy": mitigator_strategy,
                            "disinformer_strategy": disinformer_strategy,
                            "auc": calculate_auc(random_rankings)
                            - calculate_auc(exp_rankings),
                        }
                    )

                    # topic flipped
                    exp_rankings = exp_results_simplified[base_experiment_name][
                        f"{disinformer_strategy}_{mitigator_strategy}"
                    ][1]
                    random_rankings = exp_results_simplified[base_experiment_name][
                        f"{disinformer_strategy}_random"
                    ][1]
                    res.append(
                        {
                            "mitigator_fact": " ".join(bad_fact),
                            "disinformer_fact": " ".join(good_fact),
                            "mitigator_strategy": mitigator_strategy,
                            "disinformer_strategy": disinformer_strategy,
                            "auc": calculate_auc(random_rankings)
                            - calculate_auc(exp_rankings),
                        }
                    )
                except:
                    if print_err:
                        print(
                            "Failed to get auc diffs for",
                            base_experiment_name,
                            mitigator_strategy,
                            disinformer_strategy,
                        )
                        # traceback.print_exc()
                    continue
    return pd.DataFrame.from_records(res)


def get_diff_aucs(
    experiment_pairs,
    strategies,
    baseline,
    exp_diffs,
    label_map,
    print_err: bool = False,
) -> pd.DataFrame:
    res = []
    strategies_no_baseline = list(set(strategies) - set(baseline))
    for good_fact, bad_fact in experiment_pairs:
        base_experiment_name = get_base_exp_name(good_fact, bad_fact, label_map)
        for i, _ in enumerate(strategies):
            disinformer_strategy = strategies[i]
            for m, _ in enumerate(strategies_no_baseline):
                try:
                    mitigator_strategy = strategies_no_baseline[m]
                    exp_rankings = exp_diffs[base_experiment_name][
                        f"{mitigator_strategy}_{disinformer_strategy}"
                    ]
                    random_rankings = exp_diffs[base_experiment_name][
                        f"random_{disinformer_strategy}"
                    ]
                    res.append(
                        {
                            "mitigator_fact": " ".join(good_fact),
                            "disinformer_fact": " ".join(bad_fact),
                            "mitigator_strategy": mitigator_strategy,
                            "disinformer_strategy": disinformer_strategy,
                            "auc_diff": get_auc_diff_1_exp(
                                exp_rankings, random_rankings
                            ),
                        }
                    )

                    # topic flipped
                    exp_rankings = (
                        np.array(
                            exp_diffs[base_experiment_name][
                                f"{disinformer_strategy}_{mitigator_strategy}"
                            ]
                        )
                        * -1
                    )
                    random_rankings = (
                        np.array(
                            exp_diffs[base_experiment_name][
                                f"{disinformer_strategy}_random"
                            ]
                        )
                        * -1
                    )
                    res.append(
                        {
                            "mitigator_fact": " ".join(bad_fact),
                            "disinformer_fact": " ".join(good_fact),
                            "mitigator_strategy": mitigator_strategy,
                            "disinformer_strategy": disinformer_strategy,
                            "auc_diff": get_auc_diff_1_exp(
                                exp_rankings, random_rankings
                            ),
                        }
                    )
                except:
                    if print_err:
                        print(
                            "Failed to get auc diffs for",
                            base_experiment_name,
                            mitigator_strategy,
                            disinformer_strategy,
                        )
                        # traceback.print_exc()
                    continue
    return pd.DataFrame.from_records(res)


def plot_auc_diff(ax: Axes, exp_rankings, random_rankings):
    auc_diff = get_auc_diff_1_exp(exp_rankings, random_rankings)
    ax.text(0, 0, f"AUC: {auc_diff}", fontsize="medium")


def plot_random_random_diffs(ax: Axes, base_experiment_name, exp_diffs):
    random_1 = exp_diffs[base_experiment_name]["random_random_1"]

    random_2 = exp_diffs[base_experiment_name]["random_random_2"]

    ax.axhline(y=0, color="black", linestyle="-")
    ax.plot(random_1, color="blue", label="Random 1")
    ax.plot(random_2, color="red", label="Random 2")
    ax.legend(loc="lower right")
