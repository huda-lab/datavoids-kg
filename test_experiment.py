import json
from Kelpie.dataset import Dataset
from helpers.helpers import print_fact
from helpers.knowledge_graph_simulation_experiment import (
    KnowledgeGraphMitigationExperiment,
)
from helpers.budget_helpers import get_good_bad_fact_budgets
from helpers.plotting_utilities import get_base_exp_name

experiment_pairs = [
    (
        ("/m/0151w_", "/film/director/film", "/m/07kh6f3"),
        ("/m/0151w_", "/film/director/film", "/m/0h03fhx"),
    ),
    (
        ("/m/014zcr", "/film/actor/film./film/performance/film", "/m/0418wg"),
        ("/m/014zcr", "/film/actor/film./film/performance/film", "/m/07w8fz"),
    ),
]

FB15K237_PATH = "Kelpie_package/Kelpie/data/FB15k-237"
TRAIN_PATH = FB15K237_PATH + "/train.txt"
TEST_PATH = FB15K237_PATH + "/test.txt"
VALID_PATH = FB15K237_PATH + "/valid.txt"
fb15k237_dataset = Dataset(
    name="FB15k-237",
    separator="\t",
    load=True,
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    valid_path=VALID_PATH,
)
LABEL_MAP_PATH = "entity2wikidata.json"
label_map = json.load(open(LABEL_MAP_PATH, "r", encoding="utf-8"))


for good_fact, bad_fact in experiment_pairs:
    EXPERIMENT_NAME = get_base_exp_name(good_fact, bad_fact, label_map)
    NUM_ATTACK_BUDGET = 25
    print_fact(good_fact, label_map)
    print_fact(bad_fact, label_map)

    reduced_dataset_path = f"reduced_datasets/{EXPERIMENT_NAME}_reduced_dataset.txt"
    reduced_dataset = Dataset(
        name="FB15K-237",
        separator="\t",
        load=True,
        train_path=reduced_dataset_path,
        test_path=TEST_PATH,
        valid_path=VALID_PATH,
    )

    BUDGET_FILE = f"test_budgets/{EXPERIMENT_NAME}_budget.json"
    print("Budget does not exist. Creating budget.")
    get_good_bad_fact_budgets(
        reduced_dataset,
        good_fact,
        bad_fact,
        NUM_ATTACK_BUDGET,
        f"test_budgets/{EXPERIMENT_NAME}_budget_model.pt",
        BUDGET_FILE,
        "kelpie",
        "degree",
    )

# def test_random_random_multiple_rounds(
#     save_name, attack_budget, num_rounds, seed, budget_strategy
# ):
#     experiment = KnowledgeGraphMitigationExperiment(
#         save_name,
#         "FB15k-237",
#         LABEL_MAP_PATH,
#         good_fact,
#         GOOD_ENTITY,
#         BAD_ENTITY,
#         og_train_test_valid_paths=[
#             "reduced_datasets/Jackie Chan_Shanghai Knights_Rush Hour 3_reduced_dataset.txt",
#             TEST_PATH,
#             VALID_PATH,
#         ],
#         prediction_type="tail",
#         reduce_original=False,
#         size_of_original=5,
#         mode="necessary",
#         model_name="ComplEx",
#         mitigator_strategy="multi_greedy",
#         disinformer_strategy="random",
#         num_attack_budget=attack_budget,
#         num_experiments_random=num_rounds,
#         base_random_strategy_seed=seed,
#         remove_overlapping_budget_from_dv=True,
#         budget_strategy=budget_strategy,
#     )

#     experiment.run_experiment()


# # print("testing with kelpie strategy, no saved budget file")
# # test_random_random_multiple_rounds('test_1_res', 20, 1, 42, 'kelpie')

# # print("testing with neighbor strategy, no saved budget file")
# # test_random_random_multiple_rounds('test_2_res', 20, 1, 42, 'neighbor')

# print("testing with saved budget file")
# reduced_dataset = Dataset(
#     name="FB15K-237",
#     separator="\t",
#     load=True,
#     train_path="reduced_datasets/Jackie Chan_Shanghai Knights_Rush Hour 3_reduced_dataset.txt",
#     test_path=TEST_PATH,
#     valid_path=VALID_PATH,
# )
# get_good_bad_fact_budgets(
#     reduced_dataset,
#     good_fact,
#     bad_fact,
#     5,
#     "results/test_3_res/temp.pt",
#     "results/test_3_res/budget.json",
#     "kelpie",
# )

# try:
#     experiment = KnowledgeGraphMitigationExperiment(
#         "test_3_res",
#         "FB15k-237",
#         LABEL_MAP_PATH,
#         good_fact,
#         GOOD_ENTITY,
#         BAD_ENTITY,
#         og_train_test_valid_paths=[
#             "reduced_datasets/Jackie Chan_Shanghai Knights_Rush Hour 3_reduced_dataset.txt",
#             TEST_PATH,
#             VALID_PATH,
#         ],
#         prediction_type="tail",
#         reduce_original=False,
#         size_of_original=5,
#         mode="necessary",
#         model_name="ComplEx",
#         mitigator_strategy="neighbor",
#         disinformer_strategy="neighbor",
#         num_attack_budget=5,
#         num_experiments_random=1,
#         base_random_strategy_seed=42,
#         remove_overlapping_budget_from_dv=True,
#         budget_strategy="kelpie",
#         budget_file="results/test_3_res/budget.json",
#         resume_experiment=False,
#         plot_rankings=False,
#     )
#     experiment.run_experiment()
# except Exception as e:
#     print(e)
