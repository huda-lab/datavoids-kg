import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from Kelpie.dataset import Dataset

from helpers.agent import Agent
from helpers.budget_helpers import (get_good_bad_fact_budgets,
                                    load_budget_from_file,
                                    parse_budget_from_dictionary)
from helpers.constants import MAX_MIN_NORMALIZATION_COSTS, SEED
from helpers.helpers import (calculate_auc, extract_subgraph_of_kg,
                             initialize_nx_graph, print_fact,
                             print_samples_to_readable_facts)
from helpers.kelpie_models_helpers import train_complex
from helpers.strategies import (AltNeighborStrategy, ApproxGreedyStrategy,
                                GreedyStrategy, MultiObjectiveGreedyStrategy,
                                NeighborStrategy, RandomStrategy)

# fix label_map_path stuff


class KnowledgeGraphMitigationExperiment:
    def __init__(
        self,
        experiment_name: str,
        dataset_name: str,
        label_map_path: str,
        fact_in_question: tuple,
        good_entity: str,
        bad_entity: str,
        og_train_test_valid_paths: list,
        prediction_type: str = "tail",
        reduce_original: bool = True,
        size_of_original: int = 5,
        mode: str = "necessary",
        model_name: str = "ComplEx",
        mitigator_strategy: str = "random",
        disinformer_strategy: str = "random",
        num_attack_budget: int = 25,
        num_experiments_random: int = 10,
        mitigator_max_cost: int = None,
        disinformer_max_cost: int = None,
        base_random_strategy_seed: int = SEED,
        lp_model_seed: int = SEED,
        remove_overlapping_budget_from_dv: bool = True,
        save_dataset_each_round: bool = False,
        budget_file: str = None,
        budget_strategy: str = "kelpie",
        plot_rankings: bool = False,
        resume_experiment: bool = False,
        cost_type: str = "kelpie",
    ):
        self.resume_experiment = resume_experiment
        self.plot_rankings = plot_rankings
        self.budget_strategy = budget_strategy
        self.save_dataset_each_round = save_dataset_each_round
        self.remove_overlapping_budget_from_dv = remove_overlapping_budget_from_dv
        self.overlapping_budget = None
        self.budget_file = budget_file
        # set the seed
        self.base_random_strategy_seed = base_random_strategy_seed
        self.lp_model_seed = lp_model_seed
        # set the prediction type, it can be either "tail" or "head"
        self.prediction_type = prediction_type
        self.dataset_name = dataset_name
        self.graph = None
        self.good_budget = None
        self.bad_budget = None
        self.good_budget_costs = None
        self.bad_budget_costs = None
        self.experimental_dataset = None
        # right now, we choose the good entity to be the 2nd one in the ranking and the bad entity to be the 1st one in the ranking
        if not good_entity and not bad_entity:
            raise Exception("Please provide both the good and bad entities")
        # both entities should be in fact format (not sample or ID format)
        self.good_entity = good_entity
        self.bad_entity = bad_entity

        # number of times to run experiment
        if disinformer_strategy == "random" or mitigator_strategy == "random":
            self.num_epochs = num_experiments_random
        else:
            self.num_epochs = 1

        # the fact to investigate, expected in the form of [head, relation, tail]
        # the fact should contain the good entity
        self.fact_in_question = fact_in_question
        self.good_fact_in_question = fact_in_question
        self.bad_fact_in_question = (
            [fact_in_question[0], fact_in_question[1], bad_entity]
            if prediction_type == "tail"
            else [bad_entity, fact_in_question[1], fact_in_question[2]]
        )
        # this is what we use to modify the original dataset. If reduceOriginal is set to True, then we reduce the size of the
        # original dataset to size_of_original, which is  a percentage. This will be dataset we will be modifying at every step fo the simulation
        self.size_of_original = size_of_original
        self.reduce_original = reduce_original

        self.good_budget_relevance = {}
        self.bad_budget_relevance = {}
        # Original paths of the training, validation and testing datasets
        self.train_test_valid_paths = og_train_test_valid_paths

        # this is the experient name, which is used to create a directory to store the results
        self.experiment_name = experiment_name
        self.save_directory = f"./results/{self.experiment_name}"
        self.results = [{}]
        self.current_epoch = 0

        # necessary or sufficient, ive only tested necessary
        self.mode = mode
        self.model_name = model_name
        if label_map_path:
            with open(label_map_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

        # the actual dataset used during the experiment
        # could be the original dataset and could be a reduced version
        self.dataset = None

        self.num_attack_budget = num_attack_budget
        self.mitigator_strategy_type = mitigator_strategy
        self.disinformer_strategy_type = disinformer_strategy

        # initialize the strategies
        if mitigator_strategy == "neighbor":
            self.mitigator_strategy = NeighborStrategy(
                set(), {}, max_cost=mitigator_max_cost
            )
        elif mitigator_strategy == "greedy":
            self.mitigator_strategy = GreedyStrategy(
                set(),
                {},
                self.good_fact_in_question,
                self.save_directory + "/greedy_mitigator.pt",
                self.model_name,
                max_cost=mitigator_max_cost,
            )
        elif mitigator_strategy == "approx_greedy":
            self.mitigator_strategy = ApproxGreedyStrategy(
                set(), {}, {}, mitigator_max_cost
            )
        elif mitigator_strategy == "alt_neighbor":
            self.mitigator_strategy = AltNeighborStrategy(
                set(), {}, max_cost=mitigator_max_cost
            )
        elif mitigator_strategy == "multi_greedy":
            self.mitigator_strategy = MultiObjectiveGreedyStrategy(
                set(),
                {},
                {},
                self.good_fact_in_question,
                self.save_directory + "/greedy_mitigator.pt",
                self.model_name,
                MAX_MIN_NORMALIZATION_COSTS[dataset_name]['max_cost'],
                MAX_MIN_NORMALIZATION_COSTS[dataset_name]['min_cost'],
                max_cost=mitigator_max_cost,
                alpha=0.5,
            )
        else:
            self.mitigator_strategy = RandomStrategy(
                set(),
                {},
                max_cost=mitigator_max_cost,
                seed=self.base_random_strategy_seed,
            )

        if disinformer_strategy == "neighbor":
            self.disinformer_strategy = NeighborStrategy(
                set(), {}, max_cost=disinformer_max_cost
            )
        elif disinformer_strategy == "greedy":
            self.disinformer_strategy = GreedyStrategy(
                [],
                set(),
                self.bad_fact_in_question,
                self.save_directory + "/greedy_disinformer.pt",
                self.model_name,
                max_cost=disinformer_max_cost,
            )
        elif disinformer_strategy == "approx_greedy":
            self.disinformer_strategy = ApproxGreedyStrategy(
                set(), {}, {}, disinformer_max_cost
            )
        elif disinformer_strategy == "alt_neighbor":
            self.disinformer_strategy = AltNeighborStrategy(
                set(), {}, max_cost=disinformer_max_cost
            )
        elif disinformer_strategy == "multi_greedy":
            self.disinformer_strategy = MultiObjectiveGreedyStrategy(
                set(),
                {},
                {},
                self.good_fact_in_question,
                self.save_directory + "/greedy_disinformer.pt",
                self.model_name,
                MAX_MIN_NORMALIZATION_COSTS[dataset_name]['max_cost'],
                MAX_MIN_NORMALIZATION_COSTS[dataset_name]['min_cost'],
                max_cost=disinformer_max_cost,
            )
        else:
            self.disinformer_strategy = RandomStrategy(
                set(), {}, disinformer_max_cost, seed=self.base_random_strategy_seed
            )
        print("initialized strategies")
        # initialize the agents
        self.disinformer = Agent(
            self.disinformer_strategy, disinformer_strategy, agent_type="disinformer"
        )
        self.mitigator = Agent(
            self.mitigator_strategy, mitigator_strategy, agent_type="mitigator"
        )
        self.num_rounds = None
        self.cost_type = cost_type

    def _load_dataset(self):
        if self.reduce_original:
            # make a directory for the data
            os.makedirs(self.save_directory + "/data", exist_ok=True)
            print("Reducing file size of train file in the dataset",
                  self.dataset_name)
            train_dataset = Dataset(
                name=self.dataset_name,
                separator="\t",
                load=True,
                train_path=self.train_test_valid_paths[0],
                test_path=self.train_test_valid_paths[1],
                valid_path=self.train_test_valid_paths[2],
            )
            good_sample_in_question = train_dataset.fact_to_sample(
                self.fact_in_question
            )
            bad_sample_in_question = Dataset.replace_entity_in_sample(
                good_sample_in_question,
                train_dataset.get_id_for_entity_name(self.good_entity),
                train_dataset.get_id_for_entity_name(self.bad_entity),
                as_numpy=False,
            )
            # no point in reducing the size of test and valid path, since they're not used for training anyways
            extract_subgraph_of_kg(
                train_dataset,
                [good_sample_in_question, bad_sample_in_question],
                percentage_to_keep=self.size_of_original,
                save_path=self.save_directory + "/train.txt",
            )

            self.dataset = Dataset(
                name=self.dataset_name,
                separator="\t",
                load=True,
                train_path=self.save_directory + "/train.txt",
                test_path=self.train_test_valid_paths[1],
                valid_path=self.train_test_valid_paths[2],
            )

        else:
            self.dataset = Dataset(
                name=self.dataset_name,
                separator="\t",
                load=True,
                train_path=self.train_test_valid_paths[0],
                test_path=self.train_test_valid_paths[1],
                valid_path=self.train_test_valid_paths[2],
            )
        if self.disinformer.strategy_name == "neighbor":
            self.disinformer.strategy.set_dataset(
                self.dataset,
                self.dataset.get_id_for_entity_name(
                    self.bad_fact_in_question[0]),
                self.dataset.get_id_for_entity_name(
                    self.bad_fact_in_question[2]),
            )
        elif self.disinformer.strategy_name == "alt_neighbor":
            self.disinformer.strategy.set_dataset(
                self.dataset, self.dataset.fact_to_sample(
                    self.bad_fact_in_question)
            )

        if self.mitigator.strategy_name == "neighbor":
            self.mitigator.strategy.set_dataset(
                self.dataset,
                self.dataset.get_id_for_entity_name(
                    self.good_fact_in_question[0]),
                self.dataset.get_id_for_entity_name(
                    self.good_fact_in_question[2]),
            )
        elif self.mitigator.strategy_name == "alt_neighbor":
            self.mitigator.strategy.set_dataset(
                self.dataset, self.dataset.fact_to_sample(
                    self.good_fact_in_question)
            )

        # initialize graph
        self.graph = initialize_nx_graph(self.dataset)
        print("Loaded dataset")

    def _collect_initial_results_after_dv(self):
        first_model_after_data_void_save_path = (
            self.save_directory + "/first_model_after_data_void.pt"
        )
        print("Training model after creating data void\n")
        good_rank, bad_rank, _, _ = self._retrain_and_get_ranks(
            first_model_after_data_void_save_path, self.experimental_dataset
        )
        self._collect_results(
            0,
            good_rank,
            bad_rank,
        )

    def _initialize_experiment(self):
        # create a directory with the experiment name
        os.makedirs(self.save_directory, exist_ok=True)

        # load dataset
        self._load_dataset()

        print("Good fact in question:")
        print(self.good_fact_in_question)
        print_fact(self.good_fact_in_question, self.label_map)
        print("Bad fact in question:")
        print(self.bad_fact_in_question)
        print_fact(self.bad_fact_in_question, self.label_map)
        print()

        self._load_budgets()
        print("Loaded budget")

        if self.mitigator_strategy_type in ["multi_greedy", 'approx_greedy']:
            self.mitigator.add_to_attack_budget(
                self.good_budget, self.good_budget_costs, self.good_budget_relevance
            )
        else:
            self.mitigator.add_to_attack_budget(
                self.good_budget, self.good_budget_costs
            )

        if self.disinformer_strategy_type in ["multi_greedy", 'approx_greedy']:
            self.disinformer.add_to_attack_budget(
                self.bad_budget, self.bad_budget_costs, self.bad_budget_relevance
            )
        else:
            self.disinformer.add_to_attack_budget(
                self.bad_budget, self.bad_budget_costs
            )

        print("Added good and bad budget to attacker and mitigater")

        self._calculate_fact_explanation_stats()
        print("Calculated Budget Statistics")

        self.experimental_dataset = self._create_data_void(
            self.good_budget, self.bad_budget, self.overlapping_budget
        )
        print("Created data void")

        self.num_rounds = max(len(self.good_budget), len(self.bad_budget))
        print("Number of rounds for this simulation", self.num_rounds)

        if self.resume_experiment:
            if os.path.exists(self.save_directory + "/all_results.json"):
                with open(
                    self.save_directory + "/all_results.json", "r", encoding="utf-8"
                ) as file:
                    self.results = json.load(file)

                # when the experiment already completed
                if (len(self.results) >= self.num_epochs) and str(
                    self.num_rounds
                ) in self.results[self.num_epochs - 1]:
                    raise Exception("Experiment already completed")

                # when the experiment should be resumed
                # completed the last epoch
                # did not complete the last epoch
                if (len(self.results) > 0) and (self.num_epochs >= len(self.results)):
                    # if last epoch is incomplete, repeat it
                    if str(self.num_rounds) not in self.results[-1]:
                        self.current_epoch = len(self.results) - 1
                    else:
                        self.current_epoch = len(self.results)
                        self.results.append({})
                    # add results after datavoid to the current epoch (if they exist)
                    if "0" in self.results[0]:
                        self.results[self.current_epoch]["0"] = self.results[0]["0"]
                    else:
                        self._collect_initial_results_after_dv()
                    print("current epoch", self.current_epoch)
                    return

        # when the experiment had not started yet
        print("Experiment was never started")

        if len(self.results) == 0:
            self.results.append({})

        self._get_initial_ranking()
        print("Retrieved initial ranking")

        self._collect_initial_results_after_dv()

        print("Initialized experiment")

    def _get_initial_ranking(self):
        copy_dataset = copy.deepcopy(self.dataset)

        # remove the samples in question from the test dataset so they dont end up in the budget
        good_sample_in_question = copy_dataset.fact_to_sample(
            self.good_fact_in_question
        )
        bad_sample_in_question = copy_dataset.fact_to_sample(
            self.bad_fact_in_question)

        copy_dataset.remove_training_samples(
            np.array([good_sample_in_question, bad_sample_in_question])
        )

        # get ranking of facts after removing them from the data void
        good_entity_rank, bad_entity_rank, _, _ = self._retrain_and_get_ranks(
            self.save_directory + "/training_model.pt", copy_dataset
        )
        self._collect_results(
            step=-2,
            good_entity_rank=good_entity_rank,
            bad_entity_rank=bad_entity_rank,
        )

    def _retrain_and_get_ranks(
        self, model_save_path: str, dataset: Dataset, get_ranks: bool = True
    ):
        trained_model = train_complex(
            model_save_path=model_save_path,
            dataset=dataset,
            load_existing_model=False,
            seed=self.lp_model_seed,
        )

        if not get_ranks:
            return None, None, None, None
        # convert good entity and bad entity to ids
        good_entity_id = dataset.get_id_for_entity_name(self.good_entity)
        bad_entity_id = dataset.get_id_for_entity_name(self.bad_entity)
        good_sample_in_question = dataset.fact_to_sample(
            self.good_fact_in_question)

        _, _, predictions = trained_model.predict_samples(
            np.array([good_sample_in_question])
        )

        head_predictions, tail_predictions = predictions[0][0], predictions[0][1]

        if self.prediction_type == "tail":
            good_entity_rank = (
                int(np.where(tail_predictions == good_entity_id)[0][0]) + 1
            )
            bad_entity_rank = int(
                np.where(tail_predictions == bad_entity_id)[0][0]) + 1
        else:
            good_entity_rank = (
                int(np.where(head_predictions == good_entity_id)[0][0]) + 1
            )
            bad_entity_rank = int(
                np.where(head_predictions == bad_entity_id)[0][0]) + 1

        print("Good entity rank", good_entity_rank)
        print("Bad entity rank", bad_entity_rank)
        return (
            good_entity_rank,
            bad_entity_rank,
            [dataset.get_name_for_entity_id(pred)
             for pred in head_predictions[:5]],
            [dataset.get_name_for_entity_id(pred)
             for pred in tail_predictions[:5]],
        )

    def _collect_results(
        self,
        step: int,
        good_entity_rank: int = None,
        bad_entity_rank: int = None,
        good_fact_added=None,
        bad_fact_added=None,
        disinformer_spent_cost: int = 0,
        mitigator_spent_cost: int = 0,
        head_predictions: list = None,
        tail_predictions: list = None,
    ):
        #  This function collects the results at every step of the simulation.
        # and saves the results in a json file that is stored in the results directory
        print("\nCollecting results at step", step)
        # convert good entity and bad entity to ids

        self.results[self.current_epoch][str(step)] = {
            "good_fact_added": good_fact_added,
            "bad_fact_added": bad_fact_added,
            "disinformer_spent_cost": disinformer_spent_cost,
            "mitigator_spent_cost": mitigator_spent_cost,
            "good_entity_rank": good_entity_rank,
            "bad_entity_rank": bad_entity_rank,
            "head_preds": head_predictions,
            "tail_preds": tail_predictions,
        }

        with open(
            self.save_directory + "/all_results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(self.results, f)

    def _create_data_void(
        self, good_budget_samples: set, bad_budget_samples: set, overlapping_budget: set
    ):
        new_dataset = copy.deepcopy(self.dataset)
        # remove good and bad samples
        good_sample_in_question = new_dataset.fact_to_sample(
            self.good_fact_in_question)
        bad_sample_in_question = new_dataset.fact_to_sample(
            self.bad_fact_in_question)
        print("Removing:")
        print_samples_to_readable_facts(
            [good_sample_in_question, bad_sample_in_question],
            new_dataset,
            self.label_map,
        )
        print()
        new_dataset.remove_training_samples(
            np.array([good_sample_in_question, bad_sample_in_question])
        )

        # remove good and bad budget
        new_dataset.remove_training_samples(
            np.array(list(good_budget_samples)))
        new_dataset.remove_training_samples(np.array(list(bad_budget_samples)))
        print("Removing explanations:")
        print_samples_to_readable_facts(
            list(good_budget_samples), new_dataset, self.label_map
        )
        print_samples_to_readable_facts(
            list(bad_budget_samples), new_dataset, self.label_map
        )

        # remove overlapping budget
        if self.remove_overlapping_budget_from_dv:
            new_dataset.remove_training_samples(
                np.array(list(overlapping_budget)))
            print("Removing overlapping budget")

        return new_dataset

    # TODO: fix division by zero error
    def _get_avg_degree_for_samples(self, samples: list, head: bool):
        res = 0
        for sample in samples:
            if head:
                res += self.graph.degree(sample[0])
            else:
                res += self.graph.degree(sample[2])
        return res / len(samples)

    def _calculate_fact_explanation_stats(self):
        good_entity_degree = self.graph.degree(
            self.dataset.get_id_for_entity_name(self.good_entity)
        )
        bad_entity_degree = self.graph.degree(
            self.dataset.get_id_for_entity_name(self.bad_entity)
        )
        head_degree = self.graph.degree(
            self.dataset.get_id_for_entity_name(self.good_fact_in_question[0])
        )
        avg_head_degree_good_explanations = self._get_avg_degree_for_samples(
            list(self.good_budget), head=True
        )
        avg_tail_degree_good_explanations = self._get_avg_degree_for_samples(
            list(self.good_budget), head=False
        )
        avg_head_degree_bad_explanations = self._get_avg_degree_for_samples(
            list(self.bad_budget), head=True
        )
        avg_tail_degree_bad_explanations = self._get_avg_degree_for_samples(
            list(self.bad_budget), head=False
        )

        with open(self.save_directory + "/stats.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "good_fact_num_budget": len(self.good_budget),
                    "bad_fact_num_budget": len(self.bad_budget),
                    "good_entity_degree": good_entity_degree,
                    "bad_entity_degree": bad_entity_degree,
                    "head_degree": head_degree,
                    "good_explanations_degree": {
                        "head": avg_head_degree_good_explanations,
                        "tail": avg_tail_degree_good_explanations,
                    },
                    "bad_explanations_degree": {
                        "head": avg_head_degree_bad_explanations,
                        "tail": avg_tail_degree_bad_explanations,
                    },
                },
                f,
            )

    def _load_budgets(self):
        copy_dataset = copy.deepcopy(self.dataset)
        if self.budget_file is not None:
            budget_data = load_budget_from_file(self.budget_file)
        else:

            # remove the samples in question from the test dataset so they dont end up in the budget
            good_sample_in_question = copy_dataset.fact_to_sample(
                self.good_fact_in_question
            )
            bad_sample_in_question = copy_dataset.fact_to_sample(
                self.bad_fact_in_question
            )

            copy_dataset.remove_training_samples(
                np.array([good_sample_in_question, bad_sample_in_question])
            )

            budget_data = get_good_bad_fact_budgets(
                copy_dataset,
                self.good_fact_in_question,
                self.bad_fact_in_question,
                self.num_attack_budget,
                self.save_directory + "/budget_model.pt",
                self.save_directory + "/budget.json",
                self.budget_strategy,
                self.cost_type,
            )
        parsed_budget_data = parse_budget_from_dictionary(
            budget_data, copy_dataset)
        self.good_budget = parsed_budget_data["good_budget"]
        self.bad_budget = parsed_budget_data["bad_budget"]
        self.good_budget_costs = parsed_budget_data["good_budget_costs"]
        self.bad_budget_costs = parsed_budget_data["bad_budget_costs"]
        self.overlapping_budget = parsed_budget_data["overlapping_budget"]
        self.good_budget_relevance = parsed_budget_data["good_budget_relevance"]
        self.bad_budget_relevance = parsed_budget_data["bad_budget_relevance"]

    def run_experiment(self):
        self._initialize_experiment()
        good_entity_rankings, bad_entity_rankings = self._run_experiment_internal()
        self.current_epoch += 1
        if self.save_dataset_each_round:
            self.experimental_dataset.save_training_set(
                self.save_directory +
                f"/epoch_{self.current_epoch}_dataset.txt"
            )

        all_good_entity_rankings = [good_entity_rankings]
        all_bad_entity_rankings = [bad_entity_rankings]
        num_epochs_left = self.num_epochs - self.current_epoch
        for _ in range(num_epochs_left):
            self.results.append({})
            self.results[self.current_epoch]["0"] = self.results[0]["0"]
            self.experimental_dataset = self._create_data_void(
                self.good_budget, self.bad_budget, self.overlapping_budget
            )
            self.disinformer.reset_strategy()
            self.mitigator.reset_strategy()

            if self.mitigator_strategy_type in ["multi_greedy", 'approx_greedy']:
                self.mitigator.add_to_attack_budget(
                    self.good_budget, self.good_budget_costs, self.good_budget_relevance
                )
            else:
                self.mitigator.add_to_attack_budget(
                    self.good_budget, self.good_budget_costs
                )

            if self.disinformer_strategy_type in ["multi_greedy", 'approx_greedy']:
                self.disinformer.add_to_attack_budget(
                    self.bad_budget, self.bad_budget_costs, self.bad_budget_relevance
                )
            else:
                self.disinformer.add_to_attack_budget(
                    self.bad_budget, self.bad_budget_costs
                )

            (
                tmp_good_entity_rankings,
                tmp_bad_entity_rankings,
            ) = self._run_experiment_internal()
            self.current_epoch += 1

            all_good_entity_rankings.append(tmp_good_entity_rankings)
            all_bad_entity_rankings.append(tmp_bad_entity_rankings)

            # save dataset at the end of every epoch
            if self.save_dataset_each_round:
                self.experimental_dataset.save_training_set(
                    self.save_directory +
                    f"/epoch_{self.current_epoch}_dataset.txt"
                )

        print("All good entity rankings:", all_good_entity_rankings)
        print("All bad entity rankings:", all_bad_entity_rankings)
        good_entity_rankings = np.average(all_good_entity_rankings, axis=0)
        bad_entity_rankings = np.average(all_bad_entity_rankings, axis=0)

        print("Avg good entity rankings:", good_entity_rankings)
        print("Avg bad entity rankings:", bad_entity_rankings)

        if self.plot_rankings:
            self._plot(good_entity_rankings, bad_entity_rankings)

        mitigator_auc = calculate_auc(good_entity_rankings)
        disinformer_auc = calculate_auc(bad_entity_rankings)

        # save average rankings and AUC
        with open(
            self.save_directory + "/results_auc_rankings.json", "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "good_entity_rankings": list(good_entity_rankings),
                    "bad_entity_rankings": list(bad_entity_rankings),
                    "AUC_mitigator": mitigator_auc,
                    "AUC_disinformer": disinformer_auc,
                },
                f,
            )

        # save all results
        with open(
            self.save_directory + "/all_results.json", "w", encoding="utf-8"
        ) as f:
            json.dump(self.results, f)

        print("Mitigator AUC is:", mitigator_auc)
        print("Disinformer AUC is:", disinformer_auc)

    def _run_experiment_internal(self):
        print("Current epoch:", self.current_epoch)
        index = 0
        while index < self.num_rounds and (
            self.mitigator.has_valid_attack_budget()
            or self.disinformer.has_valid_attack_budget()
        ):
            print("Round number", index + 1)
            (
                disinformer_modified_dataset,
                mitigator_modified_dataset,
                mitigator_spent_cost,
            ) = (False, False, 0)
            (fact_added_mitigator, fact_added_disinformer, disinformer_spent_cost) = (
                None,
                None,
                0,
            )

            if self.disinformer.has_valid_attack_budget():
                (
                    disinformer_modified_dataset,
                    fact_added_disinformer,
                    disinformer_spent_cost,
                ) = self.disinformer.add_sample_to_dataset(
                    self.experimental_dataset, self.label_map
                )

            if self.mitigator.has_valid_attack_budget():
                (
                    mitigator_modified_dataset,
                    fact_added_mitigator,
                    mitigator_spent_cost,
                ) = self.mitigator.add_sample_to_dataset(
                    self.experimental_dataset, self.label_map
                )

            if not disinformer_modified_dataset and not mitigator_modified_dataset:
                print("Dataset was not modified")
                self._collect_results(
                    index + 1,
                    self.results[self.current_epoch][str(
                        index)]["good_entity_rank"],
                    self.results[self.current_epoch][str(
                        index)]["bad_entity_rank"],
                    fact_added_mitigator,
                    fact_added_disinformer,
                    disinformer_spent_cost,
                    mitigator_spent_cost,
                )
                index += 1
                continue

            print("Re-training model, dataset was modified")
            trained_model_path = self.save_directory + "/training_model.pt"
            (
                good_entity_rank,
                bad_entity_rank,
                head_preds,
                tail_preds,
            ) = self._retrain_and_get_ranks(
                trained_model_path, self.experimental_dataset
            )
            self._collect_results(
                step=index + 1,
                good_entity_rank=good_entity_rank,
                bad_entity_rank=bad_entity_rank,
                good_fact_added=fact_added_mitigator,
                bad_fact_added=fact_added_disinformer,
                disinformer_spent_cost=disinformer_spent_cost,
                mitigator_spent_cost=mitigator_spent_cost,
                head_predictions=head_preds,
                tail_predictions=tail_preds,
            )
            index += 1
            print()
            print("*****************")
            print()

        # TODO: collect these inside the loop instead of after
        good_entity_rankings = []
        bad_entity_rankings = []
        for index, step in enumerate(self.results[self.current_epoch].keys()):
            if int(step) >= 0:
                good_entity_rankings.append(
                    self.results[self.current_epoch][step]["good_entity_rank"]
                )
                bad_entity_rankings.append(
                    self.results[self.current_epoch][step]["bad_entity_rank"]
                )
        print("Good entity rankings", good_entity_rankings)
        print("Bad entity rankings", bad_entity_rankings)

        return good_entity_rankings, bad_entity_rankings

    def _plot(self, good_entity_rankings, bad_entity_rankings):
        # visualize the change of good entity and bad entity ranking over time
        # good entity rankings color is red
        fig, ax = plt.subplots(figsize=(4, 4))

        ax.plot(good_entity_rankings, color="blue",
                label="good entity rankings")
        # bad entity rankings color is blue
        ax.plot(bad_entity_rankings, color="red", label="bad entity rankings")
        plt.gca().invert_yaxis()

        # show a legend on the plot
        ax.legend()
        fig.savefig(self.save_directory + "/rankings.png")

        # Display a figure.
        plt.show()
