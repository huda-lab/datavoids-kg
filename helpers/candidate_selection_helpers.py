from typing import List, Optional
import json
import os
from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
from Kelpie.dataset import Dataset

from helpers.budget_helpers import get_good_bad_fact_budgets
from helpers.helpers import (NpEncoder, extract_subgraph_of_kg,
                             find_head_tail_rel, initialize_nx_graph,
                             print_entity_id, print_samples_to_readable_facts)
from helpers.kelpie_models_helpers import train_complex


def preview_samples_from_rel(rel: str, dataset: Dataset, label_map: dict):
    rel_id = dataset.get_id_for_relation_name(rel)
    res = find_head_tail_rel(dataset, rel_id=rel_id)

    print_samples_to_readable_facts(list(res)[:5], dataset, label_map)


def convert_relation_to_fn(rel):
    # change / to _
    return '_'.join(rel.split('/'))


def get_good_bad_fact_ranking(
    good_fact,
    bad_fact,
    dataset: Dataset,
    save_path,
    train_test_valid_paths,
    trained_model_save_path,
    reduced_dataset_path=None,
):
    print("Good fact:", good_fact)
    print("Bad fact:", bad_fact)

    if reduced_dataset_path:
        reduced_dataset = Dataset(
            name="",
            separator="\t",
            load=True,
            train_path=reduced_dataset_path,
            test_path=train_test_valid_paths[1],
            valid_path=train_test_valid_paths[2],
        )
    else:
        good_sample = dataset.fact_to_sample(good_fact)
        bad_sample = dataset.fact_to_sample(bad_fact)

        extract_subgraph_of_kg(
            dataset, [good_sample, bad_sample], 5, save_path=save_path
        )

        reduced_dataset = Dataset(
            name="",
            separator="\t",
            load=True,
            train_path=save_path,
            test_path=train_test_valid_paths[1],
            valid_path=train_test_valid_paths[2],
        )

    good_sample = reduced_dataset.fact_to_sample(good_fact)
    bad_sample = reduced_dataset.fact_to_sample(bad_fact)
    good_entity_id = good_sample[2]
    bad_entity_id = bad_sample[2]
    reduced_dataset.remove_training_samples([good_sample, bad_sample])
    trained_model = train_complex(
        trained_model_save_path, reduced_dataset, load_existing_model=False
    )
    _, _, predictions = trained_model.predict_samples(np.array([good_sample]))
    tail_predictions = predictions[0][1]
    return (
        np.where(tail_predictions == good_entity_id)[0][0] + 1,
        np.where(tail_predictions == bad_entity_id)[0][0] + 1,
    )


# criteria:
# large head degree
# different budgets
# similar rankings


def find_suitable_candidates(
    rel: str,
    dataset: Dataset,
    train_test_valid_paths: list,
    save_file: str,
    label_map: dict,
    reduced_dataset_path: str,
    budget_dump_file: str,
    trained_model_save_path: str,
    diff_rankings: int = 3,
    num_heads_to_test: int = 6,
    num_attack_budget: int = 25,
    overlapping_budget_threshold: int = 5,
    num_tails_per_head: int = 6,
    dataset_name: Optional[str] = None
):
    graph = initialize_nx_graph(dataset)
    rel_id = dataset.get_id_for_relation_name(rel)

    # get heads
    relevant_samples = find_head_tail_rel(dataset, rel_id=rel_id)
    head_ids = [s[0] for s in relevant_samples]
    head_ids_counter = Counter(head_ids)
    # keep only heads with multiple tails in this relation
    relevant_head_ids = {h for h, c in head_ids_counter.items() if c > 1}
    # print("Found heads:")
    # try:
    #     for h in relevant_head_ids:
    #         print_entity_id(h, dataset, label_map)
    # except:
    #     print("Could not print entity")

    if len(relevant_samples) == 0:
        print("No relevant samples found")

    facts_ranks_num_entities = []

    # check if there are existing candidates
    if os.path.exists(save_file):
        with open(save_file, "r") as inf:
            facts_ranks_num_entities = json.load(inf)

    head_degrees = list(graph.degree(relevant_head_ids))

    head_degrees.sort(reverse=True, key=lambda x: x[1])
    chosen_head_ids = head_degrees[:num_heads_to_test]
    print()
    print("We will test the following heads")
    for x in range(num_heads_to_test):
        print_entity_id(chosen_head_ids[x][0], dataset, label_map)

    print()

    estimate_duration_time(
        time_estimate_per_candidate=6,
        num_heads_to_test=num_heads_to_test,
        num_tails_per_head=num_tails_per_head,
        chosen_head_ids=chosen_head_ids,
        dataset=dataset,
        rel_id=rel_id,
        graph=graph
    )

    save_folder = "./results/generated_candidates/"

    if not dataset_name:
        dataset_name = "yago310"

    # Full path for the new directory
    dataset_save_directory = os.path.join(save_folder, dataset_name)
    # Create the new directory
    os.makedirs(dataset_save_directory, exist_ok=True)
    # Create save directory for relation name and change \ in relation name to _
    relation_save_directory = os.path.join(dataset_save_directory, convert_relation_to_fn(rel))
    os.makedirs(relation_save_directory, exist_ok=True)

    for x in range(num_heads_to_test):
        print("Testing head:")
        print_entity_id(chosen_head_ids[x][0], dataset, label_map)
        # get all tails for that head
        relevant_samples = find_head_tail_rel(
            dataset, head_id=chosen_head_ids[x][0], rel_id=rel_id
        )
        tail_ids = list({s[2] for s in relevant_samples})
        tail_degrees = list(graph.degree(tail_ids))
        tail_degrees.sort(reverse=True, key=lambda x: x[1])

        chosen_tail_ids = tail_degrees[:num_tails_per_head]
        n = len(chosen_tail_ids)

        current_candidates_number = comb(n, 2)
        print("Candidates to test for the above head:",
              current_candidates_number)
        for combo in combinations(range(n), 2):
            i, j = combo
            good_tail_id = chosen_tail_ids[i][0]
            bad_tail_id = chosen_tail_ids[j][0]

            good_fact = dataset.sample_to_fact(
                (chosen_head_ids[x][0], rel_id, good_tail_id)
            )
            bad_fact = dataset.sample_to_fact(
                (chosen_head_ids[x][0], rel_id, bad_tail_id)
            )

            print(good_fact)
            print(bad_fact)
            print()

            reduced_dataset_path = os.path.join(
                relation_save_directory, f'{good_fact}_{bad_fact}.txt')
            budget_dump_file = os.path.join(
                relation_save_directory, f'{good_fact}_{bad_fact}_budget.json')
            trained_model_save_path = os.path.join(
                relation_save_directory, f'{good_fact}_{bad_fact}_model.pt')

            if not os.path.exists(reduced_dataset_path):
                extract_subgraph_of_kg(
                    dataset, [good_fact, bad_fact], percentage_to_keep=5, save_path=reduced_dataset_path)

            print("obtained reduced dataset")

            reduced_dataset = Dataset(
                name="",
                separator="\t",
                load=True,
                train_path=reduced_dataset_path,
                test_path=train_test_valid_paths[1],
                valid_path=train_test_valid_paths[2],
            )

            budget_res = get_good_bad_fact_budgets(
                dataset=reduced_dataset,
                good_fact=good_fact,
                bad_fact=bad_fact,
                num_budget=num_attack_budget,
                trained_model_saving_path=trained_model_save_path,
                budget_dump_file=budget_dump_file,
                budget_strategy='kelpie'
            )

            print("Retrieved good and bad fact budget")
            print(budget_res)
            #  train_test_valid_paths, trained_model_save_path, reduced_dataset_path=None)

            ranks = get_good_bad_fact_ranking(
                good_fact, bad_fact, dataset, reduced_dataset_path, train_test_valid_paths, trained_model_save_path, reduced_dataset_path)

            # check overlapping budget
            # if there is more overlap than the defined treshold, you skip
            # commented out constraints i will determine this later by hand
            if len(budget_res['overlapping_budget']) > overlapping_budget_threshold:
                print("overlapping budget over threshold:",
                      len(budget_res['overlapping_budget']))
                print()
            #     continue

            # print("Budget constraints satisfied")

            print("Retrieved ranks")

            if abs(ranks[0] - ranks[1]) < diff_rankings:
                print("Rank condition satisfied")

            facts_ranks_num_entities.append(
                [good_fact, bad_fact, ranks, len(
                    budget_res["overlapping_budget"])]
            )
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(facts_ranks_num_entities, f, cls=NpEncoder)

    print()
    print()
    print()
    print()
    print()
    return facts_ranks_num_entities


def estimate_duration_time(
    time_estimate_per_candidate: int = 6,
    num_heads_to_test: int = 6,
    num_tails_per_head: int = 6,
    tail_degrees=None,
    chosen_head_ids=None,
    dataset: Dataset = None,
    rel_id=None,
    graph=None  # Replace 'GraphType' with the actual type of your graph
):
    """
    Print the total duration time for testing a given number of heads and tails.

    :param time_estimate_per_candidate: Estimated time per candidate, in hours.
    :param num_heads_to_test: Number of heads to test.
    :param num_tails_per_head: Number of tails per head.
    :param tail_degrees: Degrees of the tails (optional).
    :param chosen_head_ids: IDs of the chosen heads (optional).
    :param dataset: Dataset for the operation (optional).
    :param rel_id: Relationship ID (optional).
    :param graph: Graph data structure (optional).
    """
    if chosen_head_ids is None or dataset is None or graph is None:
        print("Missing required parameters.")
        return

    candidates_total = 0
    for x in range(num_heads_to_test):
        relevant_samples = find_head_tail_rel(
            dataset, head_id=chosen_head_ids[x][0], rel_id=rel_id)
        tail_ids = list({s[2] for s in relevant_samples})

        tail_degrees = list(graph.degree(tail_ids))
        tail_degrees.sort(reverse=True, key=lambda x: x[1])

        chosen_tail_ids = tail_degrees[:num_tails_per_head]
        n = len(chosen_tail_ids)

        current_candidates_number = comb(n, 2)
        candidates_total += current_candidates_number

    total_duration = candidates_total * time_estimate_per_candidate
    print(
        f"Estimated duration time for {candidates_total} candidates: {total_duration} hours")

# Usage example (make sure all required parameters are provided)
# estimate_duration_time(chosen_head_ids=[...], dataset=my_dataset, graph=my_graph)
