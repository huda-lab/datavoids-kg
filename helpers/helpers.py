import itertools
import json
import math
import time
import os

import networkx as nx
import numpy as np
from Kelpie.dataset import Dataset

from helpers.constants import SUPPORTED_KG_DATASETS


def calculate_auc(rankings):
    return np.trapz(rankings)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def find_head_tail_rel(dataset, head_id=None, rel_id=None, tail_id=None) -> set:
    res = set()
    for sample in dataset.train_samples_set:
        if head_id is not None:
            if head_id != sample[0]:
                continue
        if rel_id is not None:
            if rel_id != sample[1]:
                continue
        if tail_id is not None:
            if tail_id != sample[2]:
                continue
        res.add(sample)
    return res


def find_relation_in_dataset(rel, dataset: Dataset):
    rel_id = dataset.get_id_for_relation_name(rel)
    heads = set()
    for sample in dataset.train_samples_set:
        if sample[1] == rel_id and sample[0] not in heads:
            heads.add(sample[0])
    return [dataset.get_name_for_entity_id(head) for head in heads]


def find_entity_in_dataset(entity, dataset: Dataset, label_map=None):
    entity_id = dataset.get_id_for_entity_name(entity)
    for sample in dataset.train_samples_set:
        if entity_id in sample:
            try:
                print_sample(sample, dataset, label_map)
                print(sample)
            except:
                continue


def initialize_nx_graph(dataset: Dataset):
    if dataset is None:
        raise Exception("Please set the dataset first")
    graph = nx.MultiDiGraph()

    for sample in dataset.train_samples_set:
        source_node = sample[0]
        edge_type = sample[1]
        destination_node = sample[2]
        graph.add_node(source_node)
        graph.add_node(destination_node)
        graph.add_edge(source_node, destination_node,
                       key=edge_type, label=edge_type)

    return graph


def get_num_facts_good_bad(head_entity, good_tail_entity, bad_tail_entity, dataset):
    num_facts_head_entity = 0
    num_facts_good = 0
    num_facts_bad = 0
    head_entity_id = dataset.get_id_for_entity_name(head_entity)
    good_tail_entity_id = dataset.get_id_for_entity_name(good_tail_entity)
    bad_tail_entity_id = dataset.get_id_for_entity_name(bad_tail_entity)
    for sample in dataset.train_samples_set:
        if head_entity_id in sample:
            num_facts_head_entity += 1
        if good_tail_entity_id in sample:
            num_facts_good += 1
        if bad_tail_entity_id in sample:
            num_facts_bad += 1
    return num_facts_head_entity, num_facts_good, num_facts_bad


def alternate_arrays(a, b, first_list_goes_first=True):
    # Determine which is the smaller and larger array
    smaller, larger = (a, b) if len(a) < len(b) else (b, a)

    # Iterate over both arrays simultaneously until the smaller one is exhausted
    for i in range(len(smaller)):
        if first_list_goes_first:
            yield a[i]
            yield b[i]
        else:
            yield b[i]
            yield a[i]

    # Iterate over the remaining elements in the larger array
    for i in range(len(smaller), len(larger)):
        yield larger[i]


def print_fact(fact, label_map=None):
    head, relation, tail = fact[0], fact[1], fact[2]
    if label_map:
        print(label_map[head]['label'], relation, label_map[tail]['label'])
    else:
        print(fact)


def print_sample(sample, dataset, label_map):
    fact = dataset.sample_to_fact(sample)
    print_fact(fact, label_map)


def print_samples_to_readable_facts(samples, dataset: Dataset, label_map: dict):
    for sample in samples:
        print_sample(sample, dataset, label_map)


def get_readable_fact_str(fact: list, label_map: dict):
    if not label_map:
        return ''.join(fact)
    return ' '.join([label_map[fact[0]]['label'], fact[1], label_map[fact[2]]['label']])


def print_entity_id(entity_id, dataset, label_map):
    entity = dataset.get_name_for_entity_id(entity_id)
    if label_map:
        print(label_map[entity]['label'])
    else:
        print(entity)


def get_data_from_kg_name(kg_name: str):
    if kg_name not in SUPPORTED_KG_DATASETS:
        raise Exception(
            f"{kg_name} is not supported! Valid options are {', '.join(SUPPORTED_KG_DATASETS.keys())}")

    base_path = SUPPORTED_KG_DATASETS[kg_name]['base_path']
    TRAIN_PATH = f'{base_path}/train.txt'
    TEST_PATH = f'{base_path}/test.txt'
    VALID_PATH = f'{base_path}/valid.txt'
    return Dataset(name=kg_name, load=True, train_path=TRAIN_PATH, test_path=TEST_PATH, valid_path=VALID_PATH), TRAIN_PATH, TEST_PATH, VALID_PATH


def extract_subgraph_of_kg(dataset: Dataset, central_facts_to_investigate: list, percentage_to_keep: int = None, num_entries_to_keep: int = None, save_path: str = None):
    start_time = time.time()

    if percentage_to_keep is not None:
        num_entries_to_keep = math.floor(
            len(dataset.train_samples) * percentage_to_keep / 100)

    elif num_entries_to_keep is None:
        raise Exception(
            "Specify either the percentage to remove or the number of entries to remove")

    print("Reducing the dataset size from", len(
        dataset.train_samples), "to", num_entries_to_keep)

    central_samples_to_investigate = [dataset.fact_to_sample(
        fact) for fact in central_facts_to_investigate]
    new_kg = {tuple(sample) for sample in central_samples_to_investigate}
    queue = [sample for sample in central_samples_to_investigate]
    num_entries_kept = len(central_samples_to_investigate)

    num_neighbors_1, num_neighbors_2 = 0, 0

    # BFS
    while num_entries_kept < num_entries_to_keep and queue:
        # print("Generated", num_entries_kept, "entries.")
        curr_sample_1 = queue.pop(0)
        # get neighbors of head
        head_neighbors_1 = [
            list(x) for x in dataset.find_train_samples(curr_sample_1[0])]
        tail_neighbors_1 = [
            list(x) for x in dataset.find_train_samples(curr_sample_1[2])]
        all_neighbors = list(itertools.chain(
            *zip(head_neighbors_1, tail_neighbors_1)))
        num_neighbors_1 += len(all_neighbors)
        if len(queue):
            curr_sample_2 = queue.pop(0)
            head_neighbors_2 = [
                list(x) for x in dataset.find_train_samples(curr_sample_2[0])]
            tail_neighbors_2 = [
                list(x) for x in dataset.find_train_samples(curr_sample_2[2])]
            all_neighbors_2 = list(itertools.chain(
                *zip(head_neighbors_2, tail_neighbors_2)))
            num_neighbors_2 += len(all_neighbors_2)
            all_neighbors = list(itertools.chain(
                *zip(all_neighbors, all_neighbors_2)))
        for neighbor in all_neighbors:
            if num_entries_kept >= num_entries_to_keep:
                break
            if tuple(list(neighbor)) not in new_kg:
                queue.append(neighbor)
                new_kg.add(tuple(neighbor))
                num_entries_kept += 1
    print("Finished generating dataset.")
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            print("Saving in file", save_path, "...")
            with open(save_path, 'w') as file:
                for sample in new_kg:
                    fact = dataset.sample_to_fact(sample)
                    file.write('\t'.join(fact)+'\n')
            print("Saved dataset in file", save_path)
        except Exception as e:
            print("Could not save file.", e)

    end_time = time.time()
    print(f"Elapsed Time: {end_time - start_time} seconds")
    return new_kg
