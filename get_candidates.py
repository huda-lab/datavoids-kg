import json
import sys
from Kelpie.dataset import Dataset
from helpers.candidate_selection_helpers import find_suitable_candidates, convert_relation_to_fn

# rels_to_test = ['/film/director/film',
#                 '/film/actor/film./film/performance/film',
#                 '/education/educational_institution/students_graduates./education/education/student',
#                 '/organization/organization_member/member_of./organization/organization_membership/organization',
#                 '/film/film/genre']

# rels_to_test = [
#     '/award/hall_of_fame/inductees./award/hall_of_fame_induction/inductee',
#     '/sports/sports_league/teams./sports/sports_league_participation/team',
# ]

# rels_to_test = [
#     '/organization/organization_founder/organizations_founded',
#     '/influence/influence_node/influenced_by',
# ]

# rels_to_test = [
#     '/film/film/country',
#     '/award/award_nominee/award_nominations./award/award_nomination/nominated_for',
#     '/business/business_operation/industry'
# ]

rels_to_test = [
    '_derivationally_related_form',
    '_instance_hypernym',
    '_also_see',
    '_has_part',
    '_member_meronym',
    '_member_of_domain_usage',
    '_member_of_domain_region'
]

# DATASET_PATH = 'Kelpie_package/Kelpie/data/FB15k-237'
DATASET_PATH = 'Kelpie_package/Kelpie/data/WN18RR_text'
TRAIN_PATH = DATASET_PATH + '/train.txt'
TEST_PATH = DATASET_PATH + '/test.txt'
VALID_PATH = DATASET_PATH + '/valid.txt'
dataset = Dataset(name="FB15k-237",
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
file_name = convert_relation_to_fn(rel)

print("Running with args: ",
      rel, file_name,
      num_heads_to_test, num_attack_budget, overlapping_budget_threshold)

find_suitable_candidates(rel,
                         dataset,
                         train_test_valid_paths,
                         f'experiment_candidates/{file_name}.json',
                         #  label_map,
                         None,
                         f'results/candidate_results/{file_name}.txt',
                         f'results/candidate_results/{file_name}_budget.json',
                         f'results/candidate_results/{file_name}_model.pt',
                         num_heads_to_test=num_heads_to_test,
                         diff_rankings=diff_rankings,
                         num_attack_budget=num_attack_budget,
                         overlapping_budget_threshold=overlapping_budget_threshold)
