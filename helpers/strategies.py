from abc import ABC, abstractmethod
import random
import copy
from Kelpie.dataset import Dataset
from helpers.kelpie_models_helpers import train_complex
import numpy as np
import math
import networkx as nx
import heapq
from collections import defaultdict
from helpers.constants import SEED
import time
from helpers.helpers import initialize_nx_graph


class Strategy(ABC):
    """
    Abstract Strategy class that defines the interface for all concrete strategies.
    """

    def __init__(self, budget: set, budget_costs: dict, max_cost: int = math.inf):
        self.attack_budget = set()
        self.budget_costs = {}
        self.max_cost = max_cost if max_cost is not None else math.inf
        self.available_cost = max_cost if max_cost is not None else math.inf
        self.spent_cost = 0
        # all the attack facts that are within the available cost
        self.valid_attack_budget = set()
        for sample in budget:
            self.attack_budget.add(tuple(sample))
            self.budget_costs[tuple(sample)] = budget_costs[tuple(sample)]
            if budget_costs[tuple(sample)] < self.available_cost:
                self.valid_attack_budget.add(tuple(sample))

    @abstractmethod
    def next_sample(self) -> tuple:
        """
        Abstract method that must be implemented by all concrete strategies.
        """
        pass

    def reset_strategy(self):
        self.attack_budget = set()
        self.budget_costs = {}
        self.valid_attack_budget = set()
        self.available_cost = self.max_cost
        self.spent_cost = 0

    def add_to_budget(self, samples: set, budget_costs: dict):
        for sample in samples:
            self.attack_budget.add(sample)
            self.budget_costs[sample] = budget_costs[sample]
            if budget_costs[tuple(sample)] <= self.available_cost:
                self.valid_attack_budget.add(tuple(sample))

    def remove_sample(self, sample_to_remove: tuple):
        # remove the sample from the attack budget
        self.attack_budget.remove(sample_to_remove)
        self.valid_attack_budget.remove(sample_to_remove)

    def update_cost(self, sample_to_add: tuple):
        # update the available cost
        self.available_cost -= self.budget_costs[sample_to_add]
        self.spent_cost += self.budget_costs[sample_to_add]
        print("Leftover cost is", self.available_cost)

        self.remove_sample(sample_to_add)

        to_remove = set()
        # update the valid attack samples based on current available cost
        for sample in self.valid_attack_budget:
            if self.budget_costs[sample] > self.available_cost:
                to_remove.add(sample)

        self.valid_attack_budget = self.valid_attack_budget - to_remove
        return self.budget_costs[sample_to_add]


class RandomStrategy(Strategy):
    def __init__(self, budget, budget_costs, max_cost=math.inf, seed: int = SEED):
        super().__init__(budget, budget_costs, max_cost)
        self.set_seed(seed)

    def next_sample(self) -> tuple:
        if not self.attack_budget or not len(self.valid_attack_budget):
            print(f"attack budget is empty or there are no valid attack samples")
            return None

        # randomly choose a sample from the candidates and pop it from the attack budget
        print("Budget candidates:", self.valid_attack_budget)
        print("valid attack budget length:", len(self.valid_attack_budget))
        chosen_index = self.random_generator.randrange(len(self.valid_attack_budget))
        print("Chosen index:", chosen_index)
        # make sure valid attack budget is sorted to make results deterministic given a random seed
        sample_to_add = sorted(list(self.valid_attack_budget))[chosen_index]
        return sample_to_add

    def set_seed(self, seed):
        self.seed = seed
        if seed is None:
            self.random_generator = random.Random(time.time())
        else:
            self.random_generator = random.Random(seed)

    def reset_strategy(self):
        super().reset_strategy()
        if self.seed is None:
            self.set_seed(None)
        else:
            self.set_seed(self.seed + 1)


class GreedyStrategy(Strategy):
    def __init__(
        self,
        budget: set,
        budget_costs: dict,
        fact_in_question,
        model_path,
        model_name,
        prediction_type="tail",
        max_cost=math.inf,
        seed=SEED,
    ):
        super().__init__(budget, budget_costs, max_cost)
        self.fact_in_question = fact_in_question
        self.prediction_type = prediction_type
        self.model_path = model_path
        self.model_training_algo = train_complex
        if model_name == "ComplEx":
            self.model_training_algo = train_complex
        self.seed = seed

    def next_sample(self, dataset: Dataset) -> tuple:
        if not self.attack_budget or not len(self.valid_attack_budget):
            print(f"attack budget is empty")
            return None

        sample_in_question = dataset.fact_to_sample(self.fact_in_question)

        print("\nFinding best sample using greedy strategy")
        best_rank = math.inf
        best_sample = None
        test_dataset = copy.deepcopy(dataset)
        for sample in self.valid_attack_budget:
            print("Testing sample", sample)
            test_dataset.add_training_samples(np.array([sample]))
            model = self.model_training_algo(
                model_save_path=self.model_path,
                load_existing_model=False,
                dataset=test_dataset,
                seed=self.seed,
            )
            _, ranks, _ = model.predict_samples(np.array([sample_in_question]))
            if self.prediction_type == "tail":
                if ranks[0][1] < best_rank:
                    best_sample = sample
                    best_rank = ranks[0][1]
            else:
                if ranks[0][0] < best_rank:
                    best_sample = sample
                    best_rank = ranks[0][0]
            test_dataset.remove_training_sample(sample)

        return best_sample


class MultiObjectiveGreedyStrategy(Strategy):
    def __init__(
        self,
        budget: set,
        budget_costs: dict,
        budget_relevance: dict,
        fact_in_question,
        model_path,
        model_name,
        max_normalization_cost,
        min_normalization_cost,
        prediction_type="tail",
        max_cost=math.inf,
        seed=SEED,
        alpha=0.5,
    ):
        # normalize costs
        self.max_normalization_cost = max_normalization_cost
        self.min_normalization_cost = min_normalization_cost
        self.normalized_costs = self.normalize_costs(budget_costs)
        # copy budget relevance
        self.budget_relevance = {k:v for (k, v) in budget_relevance.items()}
        self.alpha = alpha
        super().__init__(budget, budget_costs, max_cost)
        self.order_to_add = self._calculate_order_to_add()

    def _calculate_order_to_add(self):
        return sorted(
            list(self.valid_attack_budget), reverse=True, key=lambda fact: (self.alpha * self.budget_relevance[tuple(fact)]) - (self.alpha * self.normalized_costs[tuple(fact)]) 
        )

    def add_to_budget(self, samples: set, budget_costs: dict, budget_relevance: dict):
        # normalize costs
        self.normalized_costs = self.normalize_costs(budget_costs)
        # copy budget relevance
        self.budget_relevance = {k:v for (k, v) in budget_relevance.items()}
        super().add_to_budget(samples, budget_costs)
        self.order_to_add = self._calculate_order_to_add()

    def normalize_costs(self, costs: dict):
        if costs:
            return {k: (v - self.min_normalization_cost) / (self.max_normalization_cost - self.min_normalization_cost) for k, v in costs.items()}
        else:
            return costs

    def reset_strategy(self):
        super().reset_strategy()
        self.normalized_costs = {}
        self.budget_relevance = {}
        self.order_to_add = []

    def next_sample(self) -> tuple:
        # score, sample = self.order_to_add.pop(0)
        # print("multi-objective greedy strategy")
        # print("selected sample", sample)
        # print("relevance of sample:", self.budget_relevance[tuple(sample)])
        # print("Cost of sample, normalized:", self.normalize_costs[tuple(sample)])
        # print("score of selected sample", score)
        # print()
        return self.order_to_add.pop(0)


class ApproxGreedyStrategy(Strategy):
    def __init__(self, budget: set, budget_costs: dict, budget_relevance:dict, max_cost: int = math.inf):
        super().__init__(budget, budget_costs, max_cost)
        self.budget_relevance = {k:v for (k, v) in budget_relevance.items()}
        self.order_to_add = self._calculate_order_to_add()

    def _calculate_order_to_add(self):
        return sorted(
            list(self.valid_attack_budget), reverse=True, key=lambda fact: self.budget_relevance[tuple(fact)]
        )

    def add_to_budget(self, samples: set, budget_costs: dict, budget_relevance: dict):
        # copy budget relevance
        self.budget_relevance = {k:v for (k, v) in budget_relevance.items()}
        super().add_to_budget(samples, budget_costs)
        self.order_to_add = self._calculate_order_to_add()

    def next_sample(self) -> tuple:
        # if not self.attack_budget or not len(self.valid_attack_budget):
        #     print(f"attack budget is empty")
        #     return None
        # sorted_valid_budget = sorted(
        #     list(self.valid_attack_budget), key=lambda fact: self.budget_costs[fact]
        # )
        return self.order_to_add.pop(0)


# TODO: Fix this strategy
# all budget facts contain the head, so the distance from the head to any of the facts will be 1 or 0
class NeighborStrategy(Strategy):
    def __init__(
        self,
        budget: set,
        budget_costs: dict,
        head_entity: int = None,
        tail_entity: int = None,
        max_cost: int = math.inf,
        original_dataset: Dataset = None,
    ):
        super().__init__(budget, budget_costs, max_cost)
        self.dataset = original_dataset
        self.distances = []
        heapq.heapify(self.distances)
        self.head_entity = head_entity
        self.tail_entity = tail_entity

        if self.dataset is not None:
            self.graph = initialize_nx_graph(self.dataset)
            self._update_distances(self.attack_budget)

    def set_dataset(
        self, original_dataset: Dataset, head_entity: int, tail_entity: int
    ):
        self.dataset = original_dataset
        self.tail_entity = tail_entity
        self.head_entity = head_entity
        self.graph = initialize_nx_graph(self.dataset)
        self._update_distances(self.attack_budget)

    def add_to_budget(self, samples: set, budget_costs: dict):
        super().add_to_budget(samples, budget_costs)
        self._update_distances(samples)

    def _update_distances(self, samples: set):
        if not self.graph:
            raise Exception("Define graph first")

        for sample in samples:
            self.graph.remove_edge(sample[0], sample[2], key=sample[1])
            try:
                if sample[0] == self.head_entity:
                    path_length = nx.shortest_path_length(
                        self.graph, sample[2], self.tail_entity
                    )
                else:
                    path_length = nx.shortest_path_length(
                        self.graph, sample[0], self.tail_entity
                    )
            except:
                path_length = math.inf
            heapq.heappush(self.distances, (path_length, sample))
            self.graph.add_edge(sample[0], sample[2], key=sample[1], label=sample[1])

    def next_sample(self) -> tuple:
        if not self.attack_budget or not len(self.valid_attack_budget):
            print(f"attack budget is empty")
            return None
        print("All samples and distances:", self.distances)
        distance, nearest_sample = heapq.heappop(self.distances)
        return nearest_sample

    def update_cost(self, sample_to_add: tuple):
        res = super().update_cost(sample_to_add)
        # update the valid attack samples based on current valid budget
        to_keep = []
        for id in range(len(self.distances)):
            d, d_sample = self.distances[id]
            if d_sample in self.valid_attack_budget:
                to_keep.append((d, d_sample))
        self.distances = to_keep

        heapq.heapify(self.distances)
        print("Updated samples and distances:", self.distances)
        return res

    def reset_strategy(self):
        super().reset_strategy()
        self.distances = []
        heapq.heapify(self.distances)


class AltNeighborStrategy(Strategy):
    def __init__(
        self,
        budget: set,
        budget_costs: dict,
        sample_in_question: str = None,
        max_cost: int = math.inf,
        original_dataset: Dataset = None,
    ):
        super().__init__(budget, budget_costs, max_cost)
        self.dataset = original_dataset
        self.distances = []
        heapq.heapify(self.distances)
        self.sample_in_question = sample_in_question

        if self.dataset is not None:
            self.graph = initialize_nx_graph(self.dataset)
            self.paths = self._update_paths()
            self._update_distances(self.attack_budget)

    def _update_paths(self):
        if not self.graph:
            raise Exception("Define graph first")
        if nx.has_path(
            self.graph, self.sample_in_question[0], self.sample_in_question[1]
        ):
            return list(
                nx.all_simple_edge_paths(
                    self.graph, self.sample_in_question[0], self.sample_in_question[1]
                )
            )

    def _update_distances(self, samples: set):
        if not self.graph or not self.paths:
            raise Exception("Define graph first")

        if not len(samples):
            return

        temp_distances = defaultdict(math.inf)
        for path in self.paths:
            for edge in path:
                edge_reversed = (edge[0], edge[2], edge[1])
                print("Edge reversed", edge_reversed)
                if edge_reversed in samples:
                    temp_distances[samples[edge_reversed]] = min(
                        len(path), temp_distances[samples[edge_reversed]]
                    )
        print("Distances", temp_distances)

        for sample in samples:
            if sample in temp_distances:
                heapq.heappush(self.distances, (temp_distances[sample], sample))
            else:
                heapq.heappush(self.distances, (math.inf, sample))

    def add_to_budget(self, samples: set, budget_costs: dict):
        super().add_to_budget(samples, budget_costs)
        self._update_distances(samples)

    def set_dataset(self, original_dataset: Dataset, sample_in_question: tuple):
        self.dataset = original_dataset
        self.sample_in_question = sample_in_question
        self.graph = initialize_nx_graph(self.dataset)
        self.paths = self._update_paths()
        self._update_distances(self.attack_budget)

    def next_sample(self) -> tuple:
        if not self.attack_budget or not len(self.valid_attack_budget):
            print(f"attack budget is empty")
            return None
        print("All samples and distances:", self.distances)
        distance, nearest_sample = heapq.heappop(self.distances)
        return nearest_sample

    def update_cost(self, sample_to_add: tuple):
        super().update_cost(sample_to_add)
        # update the valid attack samples based on current valid budget
        to_keep = []
        for id in range(len(self.distances)):
            d, d_sample = self.distances[id]
            if d_sample in self.valid_attack_budget:
                to_keep.append((d, d_sample))
        self.distances = to_keep

        heapq.heapify(self.distances)
        print("Updated samples and distances:", self.distances)

    def reset_strategy(self):
        super().reset_strategy()
        self.distances = []
        heapq.heapify(self.distances)
