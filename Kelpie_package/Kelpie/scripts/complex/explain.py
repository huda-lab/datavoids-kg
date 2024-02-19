from Kelpie.prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER
from Kelpie.link_prediction.models.model import DIMENSION, INIT_SCALE, LEARNING_RATE, OPTIMIZER_NAME, DECAY_2, \
    REGULARIZER_WEIGHT, EPOCHS, DECAY_1, \
    BATCH_SIZE, REGULARIZER_NAME
from Kelpie.link_prediction.models.complex import ComplEx
from Kelpie.criage import Criage
from Kelpie.data_poisoning import DataPoisoning
from Kelpie.kelpie import Kelpie as Kelpie
from Kelpie.dataset import Dataset
import sys
import os
import random
import numpy
import torch

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))

optimizers = ['Adagrad', 'Adam', 'SGD']
modes = ['necessary', 'sufficient']
prefilters = [TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER]


def explain(testing_facts: list, dataset: Dataset, model_path: str,  mode='sufficient', coverage=10, entities_to_convert=None, prefilter_threshold=10, relevance_threshold=None, prefilter="graph-based", dimension=1000, init=1e-3, learning_rate=0.1, optimizer='Adagrad', decay1=0.9, decay2=0.999, reg=5e-2, max_epochs=100, batch_size=1000, baseline=None, max_explanation_length=1, topk_explanations=10):

    if optimizer not in optimizers:
        raise Exception("Invalid optimizer")

    if mode not in modes:
        raise Exception("Invalid mode")

    hyperparameters = {
        DIMENSION: dimension,
        INIT_SCALE: init,
        LEARNING_RATE: learning_rate,
        OPTIMIZER_NAME: optimizer,
        DECAY_2: decay2,
        DECAY_1: decay1,
        REGULARIZER_WEIGHT: reg,
        EPOCHS: max_epochs,
        BATCH_SIZE: batch_size,
        REGULARIZER_NAME: "N3"}

    # deterministic!
    seed = 42
    torch.backends.cudnn.deterministic = True
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())

    # load the dataset and its training samples
    print("Loading dataset %s..." % dataset)

    model = ComplEx(dataset=dataset, hyperparameters=hyperparameters,
                    init_random=True)
    model.to('cuda')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    if baseline is None:
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                        relevance_threshold=relevance_threshold, max_explanation_length=max_explanation_length)
    elif baseline == "data_poisoning":
        kelpie = DataPoisoning(model=model, dataset=dataset,
                               hyperparameters=hyperparameters, prefilter_type=prefilter)
    elif baseline == "criage":
        kelpie = Criage(model=model, dataset=dataset,
                        hyperparameters=hyperparameters)
    elif baseline == "k1":
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                        relevance_threshold=relevance_threshold, max_explanation_length=1)
    else:
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                        relevance_threshold=relevance_threshold)

    testing_fact_2_entities_to_convert = None
    if mode == "sufficient" and entities_to_convert is not None:
        print("Reading entities to convert...")
        testing_fact_2_entities_to_convert = {}
        with open(entities_to_convert, "r") as entities_to_convert_file:
            entities_to_convert_lines = entities_to_convert_file.readlines()
            i = 0
            while i < len(entities_to_convert_lines):
                cur_head, cur_rel, cur_name = entities_to_convert_lines[i].strip().split(
                    ";")
                assert [cur_head, cur_rel, cur_name] in testing_facts
                cur_entities_to_convert = entities_to_convert_lines[i + 1].strip().split(
                    ",")
                testing_fact_2_entities_to_convert[(
                    cur_head, cur_rel, cur_name)] = cur_entities_to_convert
                i += 3

    for i, fact in enumerate(testing_facts):
        head, relation, tail = fact
        print("Explaining fact " + str(i) + " on " + str(
            len(testing_facts)) + ": <" + head + "," + relation + "," + tail + ">")
        head_id, relation_id, tail_id = dataset.get_id_for_entity_name(head), \
            dataset.get_id_for_relation_name(relation), \
            dataset.get_id_for_entity_name(tail)
        sample_to_explain = (head_id, relation_id, tail_id)

        # sufficient mode
        if mode == "sufficient":
            entities_to_convert_ids = None if testing_fact_2_entities_to_convert is None \
                else [dataset.entity_name_2_id[x] for x in testing_fact_2_entities_to_convert[(head, relation, tail)]]

            rule_samples_with_relevance, \
                entities_to_convert_ids = kelpie.explain_sufficient(sample_to_explain=sample_to_explain,
                                                                    perspective="head",
                                                                    num_promising_samples=prefilter_threshold,
                                                                    num_entities_to_convert=coverage,
                                                                    entities_to_convert=entities_to_convert_ids)
            return rule_samples_with_relevance

        # necessary mode
        else:
            rule_samples_with_relevance = kelpie.explain_necessary(sample_to_explain=sample_to_explain,
                                                                   perspective="head",
                                                                   num_promising_samples=prefilter_threshold, top_k=topk_explanations)
            return rule_samples_with_relevance
