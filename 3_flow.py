
# =based it off this. also, in the previous step we created the smaller txt file %
"""

Ben Affleck director Argo The Town (input_1.txt)
/m/0151w_-/film/director/film-/m/07kh6f3
/m/0151w_-/film/director/film-/m/0h03fhx

Steven Spielberg director Saving Private Ryan Amistad (input_2.txt) -- flow 2 didnt produce exact same results
/m/06pj8-/film/director/film-/m/0260bz
/m/06pj8-/film/director/film-/m/07024

George Clooney actor Good Night, and Good Luck. Ocean’s Twelve (input_7.txt)
/m/014zcr-/film/actor/film./film/performance/film-/m/0418wg
/m/014zcr-/film/actor/film./film/performance/film-/m/07w8fz

there is no producer in reems stuff, only actor film other tha


Ben Affleck producer Argo Pearl Harbor (input_10)
/m/0151w_-/film/actor/film./film/performance/film-/m/0pc62
/m/0151w_-/film/actor/film./film/performance/film-/m/0h03fhx

"""

import json
import os
import sys
import traceback
import argparse
from Kelpie.dataset import Dataset
from helpers.helpers import extract_subgraph_of_kg, print_fact
from helpers.knowledge_graph_simulation_experiment import (
    KnowledgeGraphMitigationExperiment,
)
from helpers.candidate_selection_helpers import tuple_to_filename, convert_relation_to_fn
from helpers.constants import SEED
from helpers.budget_helpers import get_good_bad_fact_budgets
from helpers.helpers import get_data_from_kg_name

def main(kg_name, good_fact_str, bad_fact_str, num_attack_budget, part, num_random_reps, regenerate_files=False):
    dataset, TRAIN_PATH, TEST_PATH, VALID_PATH, LABEL_MAP_PATH = get_data_from_kg_name(kg_name)
    LABEL_MAP = json.load(open(LABEL_MAP_PATH))
    
    good_fact = good_fact_str.split("-")
    bad_fact = bad_fact_str.split("-")

    if not good_fact or not bad_fact:
        raise ValueError("Issue reading input")

    print("Now testing")
    print_fact(good_fact, LABEL_MAP)
    print("versus")
    print_fact(bad_fact, LABEL_MAP)

    EXPERIMENT_NAME = f"{LABEL_MAP[good_fact[0]]['label']}_{LABEL_MAP[good_fact[2]]['label']}_{LABEL_MAP[bad_fact[2]]['label']}"
    REMOVE_OVERLAPPING_BUDGET_FROM_DV = True
    relation = good_fact[1]
    EXP_FOLDER = f"./results/simulations/{kg_name}/"
    BUDGET_FILE = f"./results/generated_candidates/{kg_name}/{convert_relation_to_fn(relation)}/{tuple_to_filename(good_fact)}_{tuple_to_filename(bad_fact)}_budget.json"
    reduced_dataset_path = f"./results/generated_candidates/{kg_name}/{convert_relation_to_fn(relation)}/{tuple_to_filename(good_fact)}_{tuple_to_filename(bad_fact)}.txt"
    trained_model_save_path = f"./results/generated_candidates/{kg_name}/{convert_relation_to_fn(relation)}/{tuple_to_filename(good_fact)}_{tuple_to_filename(bad_fact)}_model.pt"

    print(reduced_dataset_path)

    if regenerate_files:
        print("Reduced dataset does not exist. Creating it!")
        res = extract_subgraph_of_kg(
            dataset, [good_fact, bad_fact], 5, save_path=reduced_dataset_path
        )
    else: 
        if not os.path.exists(reduced_dataset_path):
            print("Reduced dataset does not exist. Run flow 2 first or activate regenerate_files files")
            return 


    reduced_dataset = Dataset(
        name=kg_name,
        separator="\t",
        load=True,
        train_path=reduced_dataset_path,
        test_path=TEST_PATH,
        valid_path=VALID_PATH,
    )

    if regenerate_files:
        print("Budget does not exist. Creating budget.")
        get_good_bad_fact_budgets(
            reduced_dataset,
            good_fact,
            bad_fact,
            num_attack_budget,
            trained_model_save_path,
            BUDGET_FILE,
            "kelpie",
            "degree",
        )
    else: 
        if not os.path.exists(BUDGET_FILE):
            print("Budget file does not exist! run flow 2 again or activate regenrate_files flag!")
            return
    
    good_entity = good_fact[2]
    bad_entity = bad_fact[2]
    print("good entity:", LABEL_MAP[good_entity]["label"])
    print("bad entity:", LABEL_MAP[bad_entity]["label"])

    STRATEGIES = ['approx_greedy', 'multi_greedy', 'random']

    if part == 1:
        mitigator_strategies = ['approx_greedy', 'random']
    else:
        mitigator_strategies = ['multi_greedy']

    for mitigator_strategy in mitigator_strategies:
        for disinformer_strategy in STRATEGIES:
            print(f"Running {mitigator_strategy}-{disinformer_strategy} experiment")
            try:
                experiment = KnowledgeGraphMitigationExperiment(
                    f"{EXP_FOLDER}/{EXPERIMENT_NAME}_{mitigator_strategy}_{disinformer_strategy}",
                    kg_name,
                    LABEL_MAP_PATH,
                    good_fact,
                    good_entity,
                    bad_entity,
                    og_train_test_valid_paths=[reduced_dataset_path, TEST_PATH, VALID_PATH],
                    prediction_type="tail",
                    reduce_original=False,
                    mode="necessary",
                    model_name="ComplEx",
                    mitigator_strategy=mitigator_strategy,
                    disinformer_strategy=disinformer_strategy,
                    num_attack_budget=num_attack_budget,
                    num_experiments_random=num_random_reps,
                    base_random_strategy_seed=SEED,
                    lp_model_seed=SEED,
                    remove_overlapping_budget_from_dv=REMOVE_OVERLAPPING_BUDGET_FROM_DV,
                    save_dataset_each_round=False,
                    budget_file=BUDGET_FILE,
                    resume_experiment=False,
                    cost_type="degree",
                )
                experiment.run_experiment()
            except Exception as e:
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process KG dataset and relation parameters.')
    parser.add_argument('--kg_name', type=str, required=True,
                        help='Name of the knowledge graph dataset')
    parser.add_argument('--good_fact', type=str, required=True,
                        help='Good fact to test, format: entity-relation-entity')
    parser.add_argument('--bad_fact', type=str, required=True,
                        help='Bad fact to test, format: entity-relation-entity')
    parser.add_argument('--num_attack_budget', type=int,
                        required=True, help='Number of attack budget')
    parser.add_argument('--part', type=int, required=True,
                        help='Part number to determine mitigator strategies')
    parser.add_argument('--num_random_reps', type=int, required=True,
                        help='Number of random reps')
    parser.add_argument('--regenerate_files', action='store_true', 
                        help='Should you regenerate the budget file and the small txt db?')

    args = parser.parse_args()
    main(args.kg_name, args.good_fact, args.bad_fact,
         args.num_attack_budget, args.part, args.num_random_reps, args.regenerate_files)
