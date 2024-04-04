import argparse
import json

from helpers.candidate_selection_helpers import (convert_relation_to_fn,
                                                 find_suitable_candidates)
from helpers.helpers import get_data_from_kg_name


def main(kg_name, rels_to_test, num_heads_to_test, num_attack_budget, overlapping_budget_threshold, diff_rankings):
    # Get the dataset and paths
    dataset, TRAIN_PATH, TEST_PATH, VALID_PATH = get_data_from_kg_name(kg_name)
    train_test_valid_paths = [TRAIN_PATH, TEST_PATH, VALID_PATH]

    for rel_id in rels_to_test:
        rel = rel_id
        file_name = convert_relation_to_fn(rel)

        print("Running with args: ", rel, file_name, num_heads_to_test,
              num_attack_budget, overlapping_budget_threshold, diff_rankings)

        # Finding suitable candidates
        find_suitable_candidates(rel, dataset, train_test_valid_paths,
                                 f'experiment_candidates/{file_name}.json', None,
                                 f'results/candidate_results/{file_name}.txt',
                                 f'results/candidate_results/{file_name}_budget.json',
                                 f'results/candidate_results/{file_name}_model.pt',
                                 num_heads_to_test=num_heads_to_test,
                                 diff_rankings=diff_rankings,
                                 num_attack_budget=num_attack_budget,
                                 overlapping_budget_threshold=overlapping_budget_threshold,
                                 dataset_name=kg_name
                                 )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process KG dataset and relation parameters.')
    parser.add_argument('--kg_name', type=str, required=True,
                        help='Name of the knowledge graph dataset')
    parser.add_argument('--rels_to_test', nargs='+',
                        required=True, help='Relations to test')
    parser.add_argument('--num_heads_to_test', type=int,
                        required=True, help='Number of heads to test')
    parser.add_argument('--num_attack_budget', type=int,
                        required=True, help='Number of attack budget')
    parser.add_argument('--overlapping_budget_threshold', type=int,
                        required=True, help='Overlapping budget threshold')
    parser.add_argument('--diff_rankings', type=int,
                        required=True, help='Different rankings')

    args = parser.parse_args()
    main(args.kg_name, args.rels_to_test, args.num_heads_to_test,
         args.num_attack_budget, args.overlapping_budget_threshold, args.diff_rankings)
