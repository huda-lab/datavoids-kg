import json
import sys
from Kelpie.dataset import Dataset
from helpers.candidate_selection_helpers import find_suitable_candidates, convert_relation_to_fn

# we are doing the following relation types
# 1 - n
# n - n
# directed could have gone here but too many directed type stuff in FB15K-237
# rels_to_test = [
#     'isknownfor', # 1-n
#     'wrotemusicfor', # 1-n
#     'edited', # 1-n
#     'isaffiliatedto', #n-n
#     'playsfor', # n-n
#     'haswonprize' # n-n
# ]
rels_to_test = [
    "ispoliticianof",
    "haswebsite",
    "iscitizenof"
]


DATASET_NAME = "yago310"
DATASET_PATH = 'Kelpie_package/Kelpie/data/YAGO3-10'
TRAIN_PATH = DATASET_PATH + '/train.txt'
TEST_PATH = DATASET_PATH + '/test.txt'
VALID_PATH = DATASET_PATH + '/valid.txt'
dataset = Dataset(name=DATASET_NAME,
                  load=True,
                  train_path=TRAIN_PATH,
                  test_path=TEST_PATH,
                  valid_path=VALID_PATH)
train_test_valid_paths = [TRAIN_PATH, TEST_PATH, VALID_PATH]


rel_id = int(sys.argv[1])
rel = rels_to_test[rel_id-1]

num_heads_to_test = int(sys.argv[2])
num_attack_budget = int(sys.argv[3])
overlapping_budget_threshold = int(sys.argv[4])
diff_rankings = int(sys.argv[5])
num_tails_per_head = int(sys.argv[6])
file_name = convert_relation_to_fn(rel)

print("Running with args: ",
      rel, file_name,
      num_heads_to_test, num_attack_budget, overlapping_budget_threshold, num_tails_per_head)


find_suitable_candidates(rel = rel,
                         dataset = dataset,
                         train_test_valid_paths = train_test_valid_paths,
                        save_file = f'experiment_candidates/{file_name}.json',
                         label_map = None,
                         reduced_dataset_path = f'results/candidate_results/{file_name}.txt',
                         budget_dump_file = f'results/candidate_results/{file_name}_budget.json',
                         trained_model_save_path = f'results/candidate_results/{file_name}_model.pt',
                         num_heads_to_test=num_heads_to_test,
                         diff_rankings=diff_rankings,
                         num_tails_per_head = num_tails_per_head,
                         num_attack_budget=num_attack_budget,
                         overlapping_budget_threshold=overlapping_budget_threshold, 
                         dataset_name = DATASET_NAME,
                         save_folder = "./results")