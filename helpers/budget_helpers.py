import copy
import json
import math
from functools import cmp_to_key

import networkx as nx
import numpy as np
from Kelpie.dataset import Dataset
from Kelpie.scripts.complex.explain import explain

from helpers.constants import SEED
from helpers.helpers import initialize_nx_graph, print_entity_id
from helpers.kelpie_models_helpers import train_complex


def load_budget_from_file(fn: str):
    with open(fn, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_budget_from_dictionary(budget_data: dict, dataset: Dataset):
    res = {}
    res["good_budget"] = [dataset.fact_to_sample(
        f) for f in budget_data["good_budget"]]
    res["bad_budget"] = [dataset.fact_to_sample(
        f) for f in budget_data["bad_budget"]]
    res["overlapping_budget"] = [
        dataset.fact_to_sample(f) for f in budget_data["overlapping_budget"]
    ]

    def handle_key_split(s):
        parts = s.split("-")
        try:
            if len(parts) == 3:
                return parts
            else:
                # Assuming the format is like 'sino-french_war-happenedin-china'
                # We join the first two parts back together
                return [parts[0] + '-' + parts[1], parts[2], parts[3]]
        except ValueError:
            print(f"Error processing key: {s}")
            raise

    res["good_budget_costs"] = {}
    for s, c in budget_data["good_budget_costs"].items():
        res["good_budget_costs"][dataset.fact_to_sample(
            handle_key_split(s))] = c

    res["bad_budget_costs"] = {}
    for s, c in budget_data["bad_budget_costs"].items():
        res["bad_budget_costs"][dataset.fact_to_sample(
            handle_key_split(s))] = c

    res["good_budget_relevance"] = {}
    for s, c in budget_data["good_budget_relevance"].items():
        res["good_budget_relevance"][dataset.fact_to_sample(
            handle_key_split(s))] = c

    res["bad_budget_relevance"] = {}
    for s, c in budget_data["bad_budget_relevance"].items():
        res["bad_budget_relevance"][dataset.fact_to_sample(
            handle_key_split(s))] = c

    return res


def compare_explanations(a, b):
    if a[1] < b[1]:
        return -1
    elif a[1] > b[1]:
        return 1
    else:
        return 0


def get_budget(
    fact,
    dataset: Dataset,
    prediction_type: str,
    model_name: str,
    trained_model_save_path: str,
    num_budget: int,
    cost_type: str = "kelpie",
    graph=None,
) -> (set, dict, dict, dict):

    print("Cost type:", cost_type)
    # get Kelpie explanations
    if prediction_type == "tail":
        if model_name == "ComplEx":
            target_sample = dataset.fact_to_sample(fact)
            budget_costs = {}
            budget_degrees = {}
            budget = set()
            train_complex(
                model_save_path=trained_model_save_path,
                dataset=dataset,
                load_existing_model=False,
                seed=SEED,
            )
            print("trained model")
            explanations = explain(
                testing_facts=[fact],
                dataset=dataset,
                model_path=trained_model_save_path,
                mode="necessary",
                max_explanation_length=1,
                topk_explanations=num_budget,
                # prefilter_threshold=num_budget + 10,
                prefilter_threshold=50,
                relevance_threshold=1,
            )
            explanations_sorted = sorted(
                explanations, key=cmp_to_key(compare_explanations)
            )

            curr_cost = 1
            for sample, _ in explanations_sorted:
                budget.add(tuple(sample[0]))
                if cost_type == "kelpie":
                    budget_costs[tuple(sample[0])] = curr_cost
                else:
                    if sample[0][0] == target_sample[0]:
                        print("Getting degree of tail")
                        budget_degrees[tuple(sample[0])] = graph.degree(
                            sample[0][1])
                        budget_costs[tuple(sample[0])] = graph.degree(
                            sample[0][1])
                    else:
                        print("Getting degree of head")
                        budget_degrees[tuple(sample[0])] = graph.degree(
                            sample[0][0])
                        budget_costs[tuple(sample[0])] = graph.degree(
                            sample[0][0])

                curr_cost += 1

            budget_relevance = {x[0][0]: x[1] for x in explanations_sorted}
            return budget, budget_costs, budget_relevance, budget_degrees
        else:
            raise Exception("Only ComplEx is supported at this moment.")
    else:
        raise Exception("Only tail prediction is supported at the moment.")


def get_budget_neighbors(
    fact: tuple, copy_dataset: Dataset, copy_dataset_graph, num_attack_budget: int
):
    sample = copy_dataset.fact_to_sample(fact)
    # run BFS starting at tail
    edges = list(nx.edge_bfs(copy_dataset_graph, sample[2]))
    edges_clean = [(h, r, t) for (h, t, r) in edges[:num_attack_budget]]
    return set(edges_clean), {edges_clean[i]: i + 1 for i in range(num_attack_budget)}


def get_good_bad_fact_budgets(
    dataset: Dataset,
    good_fact: tuple,
    bad_fact: tuple,
    num_budget: int,
    trained_model_saving_path: str,
    budget_dump_file: str,
    budget_strategy: str,
    cost_type: str = "kelpie",
) -> dict:

    print("Dataset name", dataset.name)
    print("num of budget", num_budget)
    copy_dataset = copy.deepcopy(dataset)

    graph = None
    if cost_type == "degree":
        graph = initialize_nx_graph(copy_dataset)

    # make sure the good and bad facts are not in the dataset
    good_sample_in_question = copy_dataset.fact_to_sample(good_fact)
    bad_sample_in_question = copy_dataset.fact_to_sample(bad_fact)
    copy_dataset.remove_training_samples(
        np.array([good_sample_in_question, bad_sample_in_question])
    )

    # get good and bad budget
    if budget_strategy == "kelpie":
        print("getting good budget with kelpie")
        (
            good_budget,
            good_budget_costs,
            good_budget_relevance,
            good_budget_degrees,
        ) = get_budget(
            good_fact,
            copy_dataset,
            "tail",
            "ComplEx",
            trained_model_saving_path,
            num_budget,
            cost_type,
            graph,
        )
        print("getting bad budget with kelpie")
        (
            bad_budget,
            bad_budget_costs,
            bad_budget_relevance,
            bad_budget_degrees,
        ) = get_budget(
            bad_fact,
            copy_dataset,
            "tail",
            "ComplEx",
            trained_model_saving_path,
            num_budget,
            cost_type,
            graph,
        )
    else:
        copy_graph = initialize_nx_graph(copy_dataset)
        good_budget, good_budget_costs = get_budget_neighbors(
            good_fact, copy_dataset, copy_graph, num_budget
        )
        bad_budget, bad_budget_costs = get_budget_neighbors(
            bad_fact, copy_dataset, copy_graph, num_budget
        )

    print("good budget length", len(good_budget), len(good_budget_costs))
    print("bad budget length", len(bad_budget), len(bad_budget_costs))

    # get overlapping budget and remove it from both good and bad budget
    overlapping_budget = good_budget.intersection(bad_budget)

    print("overlapping budet", len(overlapping_budget))
    good_budget = good_budget.difference(overlapping_budget)
    print("new good budget length after removing overlapping", len(good_budget))
    bad_budget = bad_budget.difference(overlapping_budget)
    print("new bad budget length after removing overlapping", len(bad_budget))

    # remove overlapping budget from good and bad budget costs
    for sample in overlapping_budget:
        del good_budget_costs[sample]
        del bad_budget_costs[sample]
        del good_budget_relevance[sample]
        del bad_budget_relevance[sample]
        del good_budget_degrees[sample]
        del bad_budget_degrees[sample]

    # update the remainder of the good and bad budgetcosts
    if cost_type == "kelpie":
        sorted_good_budget = list(
            dict(sorted(good_budget_costs.items(), key=lambda x: x[1])).keys()
        )
        for i in range(1, len(sorted_good_budget) + 1):
            good_budget_costs[sorted_good_budget[i - 1]] = i

        sorted_bad_budget = list(
            dict(sorted(bad_budget_costs.items(), key=lambda x: x[1])).keys()
        )
        for i in range(1, len(sorted_bad_budget) + 1):
            bad_budget_costs[sorted_bad_budget[i - 1]] = i

    res_dict = {
        "good_budget": [copy_dataset.sample_to_fact(s) for s in good_budget],
        "good_budget_costs": {
            "-".join(list(copy_dataset.sample_to_fact(s))): c
            for s, c in good_budget_costs.items()
        },
        "bad_budget": [copy_dataset.sample_to_fact(s) for s in bad_budget],
        "bad_budget_costs": {
            "-".join(list(copy_dataset.sample_to_fact(s))): c
            for s, c in bad_budget_costs.items()
        },
        "good_budget_degrees": {
            "-".join(list(copy_dataset.sample_to_fact(s))): c
            for s, c in good_budget_degrees.items()
        },
        "bad_budget_degrees": {
            "-".join(list(copy_dataset.sample_to_fact(s))): c
            for s, c in bad_budget_degrees.items()
        },
        "overlapping_budget": [
            copy_dataset.sample_to_fact(s) for s in overlapping_budget
        ],
    }

    print("good budget relevance")
    print(good_budget_relevance)
    if budget_strategy == "kelpie":
        res_dict["good_budget_relevance"] = {
            "-".join(list(copy_dataset.sample_to_fact(s))): c
            for s, c in good_budget_relevance.items()
        }
        res_dict["bad_budget_relevance"] = {
            "-".join(list(copy_dataset.sample_to_fact(s))): c
            for s, c in bad_budget_relevance.items()
        }

    # save results
    with open(budget_dump_file, "w", encoding="utf-8") as f:
        json.dump(res_dict, f)

    return res_dict
