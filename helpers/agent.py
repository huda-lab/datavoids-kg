from Kelpie.dataset import Dataset
from helpers.helpers import print_sample
import numpy as np
from helpers.strategies import Strategy
# the following class is used to to keep track of the agents' attack budget and the samples they have added to the dataset
# this class also allows the agents to modify the dataset interfacte through the modify_dataset function.
# an agent could be a disinformer or a mitigator, but this is just arbitrary and done for the sake of the simulation.


class Agent:
    def __init__(self, strategy: Strategy, strategy_name, agent_type="disinformer"):
        self.agent_type = agent_type
        self.strategy_name = strategy_name
        self.strategy = strategy

    def add_to_attack_budget(self, new_samples: set, sample_costs: dict, budget_relevance:dict=None):
        if self.strategy_name == 'multi_greedy' or self.strategy_name == 'approx_greedy':
            self.strategy.add_to_budget(new_samples, sample_costs, budget_relevance)
        else:
            self.strategy.add_to_budget(new_samples, sample_costs)


    def reset_strategy(self):
        self.strategy.reset_strategy()

    def has_valid_attack_budget(self):
        print("Valid budget left:", self.agent_type,
              len(self.strategy.valid_attack_budget))
        print()
        return len(self.strategy.attack_budget) and len(self.strategy.valid_attack_budget)

    def add_sample_to_dataset(self, dataset: Dataset, label_map) -> (bool, tuple, int):
        if self.strategy_name in ['greedy']:
            sample_to_add = self.strategy.next_sample(
                dataset)
        else:
            sample_to_add = self.strategy.next_sample()

        if not sample_to_add:
            return False, None, 0

        print(f"{self.agent_type} wants to add the following sample to the dataset")
        print_sample(sample_to_add, dataset, label_map)

        # check if dataset already contains the sample
        if any(np.array_equal(sample_to_add, big_arr) for big_arr in dataset.train_samples):
            print("Dataset already contains the sample")
            self.strategy.remove_sample(sample_to_add)
            return False, dataset.sample_to_fact(sample_to_add), 0

        dataset.add_training_samples(np.array([sample_to_add]))
        # update cost
        spent_cost = self.strategy.update_cost(sample_to_add)
        return True, dataset.sample_to_fact(sample_to_add), spent_cost
