import time

import numpy
import torch
from Kelpie.dataset import Dataset
from Kelpie.kelpie import Kelpie as Kelpie
from Kelpie.link_prediction.evaluation.evaluation import Evaluator
from Kelpie.link_prediction.models.complex import ComplEx
from Kelpie.link_prediction.models.model import (BATCH_SIZE, DECAY_1, DECAY_2,
                                                 DIMENSION, EPOCHS, INIT_SCALE,
                                                 LEARNING_RATE, OPTIMIZER_NAME,
                                                 REGULARIZER_NAME,
                                                 REGULARIZER_WEIGHT)
from Kelpie.link_prediction.models.transe import TransE
from Kelpie.link_prediction.optimization.multiclass_nll_optimizer import \
    MultiClassNLLOptimizer

from helpers.constants import SEED


def train_complex(model_save_path, dataset, dataset_name=None, hyperparameters=None, load_existing_model=False, existing_model_path=None, evaluate_model=False, seed=SEED):
    if seed is None:
        seed = time.time()
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)

    if not dataset:
        dataset = Dataset(name=dataset_name, separator="\t", load=True)

    if not hyperparameters:
        hyperparameters = {DIMENSION: 1000,
                           INIT_SCALE: 1e-3,
                           OPTIMIZER_NAME: 'Adagrad',
                           BATCH_SIZE: 1000,
                           EPOCHS: 100,
                           LEARNING_RATE: 0.1,
                           DECAY_1: 0.9,
                           DECAY_2: 0.999,
                           REGULARIZER_NAME: 'N3',
                           REGULARIZER_WEIGHT: 5e-2}

    if load_existing_model:
        model = ComplEx(dataset=dataset, hyperparameters=hyperparameters,
                        init_random=True)  # type: TransE
        model.to('cuda')
        model.load_state_dict(torch.load(existing_model_path))
        return model

    model = ComplEx(dataset=dataset, hyperparameters=hyperparameters,
                    init_random=True)   # type: ComplEx
    model.to('cuda')

    optimizer = MultiClassNLLOptimizer(model=model,
                                       hyperparameters=hyperparameters)

    optimizer.train(train_samples=dataset.train_samples,
                    save_path=model_save_path,
                    evaluate_every=-1,
                    valid_samples=dataset.valid_samples)

    if evaluate_model:
        print("Evaluating model...")
        mrr, h1, h10, mr = Evaluator(model=model).evaluate(
            samples=dataset.test_samples, write_output=False)
        print("\tTest Hits@1: %f" % h1)
        print("\tTest Hits@10: %f" % h10)
        print("\tTest Mean Reciprocal Rank: %f" % mrr)
        print("\tTest Mean Rank: %f" % mr)
    return model
