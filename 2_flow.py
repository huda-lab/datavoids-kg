import argparse
import json
from Kelpie.dataset import Dataset
from helpers.candidate_selection_helpers import find_suitable_candidates, convert_relation_to_fn

def main(kg_name, rels_to_test, rel_id, num_heads_to_test, num_attack_budget, overlapping_budget_threshold, diff_rankings):
    # Dataset paths
    DATASET_PATH = f'Kelpie_package/Kelpie/data/{kg_name}'
    TRAIN_PATH = f'{DATASET_PATH}/train.txt'
    TEST_PATH = f'{DATASET_PATH}/test.txt'
    VALID_PATH = f'{DATASET_PATH}/valid.txt'
    
    # Initialize the dataset
    dataset = Dataset(name=kg_name, load=True, train_path=TRAIN_PATH, test_path=TEST_PATH, valid_path=VALID_PATH)
    train_test_valid_paths = [TRAIN_PATH, TEST_PATH, VALID_PATH]

    rel = rels_to_test[rel_id - 1]
    file_name = convert_relation_to_fn(rel)

    print("Running with args: ", rel, file_name, num_heads_to_test, num_attack_budget, overlapping_budget_threshold, diff_rankings)

    # Finding suitable candidates
    find_suitable_candidates(rel, dataset, train_test_valid_paths,
                             f'experiment_candidates/{file_name}.json', None,
                             f'results/candidate_results/{file_name}.txt',
                             f'results/candidate_results/{file_name}_budget.json',
                             f'results/candidate_results/{file_name}_model.pt',
                             num_heads_to_test=num_heads_to_test,
                             diff_rankings=diff_rankings,
                             num_attack_budget=num_attack_budget,
                             overlapping_budget_threshold=overlapping_budget_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process KG and relation parameters.')
    parser.add_argument('kg_name', type=str, help='Name of the knowledge graph')
    parser.add_argument('rels_to_test', nargs='+', help='List of relations to test')
    parser.add_argument('rel_id', type=int, help='ID of the relation to test')
    parser.add_argument('num_heads_to_test', type=int, help='Number of heads to test')
    parser.add_argument('num_attack_budget', type=int, help='Number of attack budget')
    parser.add_argument('overlapping_budget_threshold', type=int, help='Overlapping budget threshold')
    parser.add_argument('diff_rankings', type=int, help='Different rankings')

    args = parser.parse_args()
    main(args.kg_name, args.rels_to_test, args.rel_id, args.num_heads_to_test, args.num_attack_budget, args.overlapping_budget_threshold, args.diff_rankings)
