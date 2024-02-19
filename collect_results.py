import json
import os
from helpers.plotting_utilities import get_base_exp_name

LABEL_MAP_PATH = 'entity2wikidata.json'
LABEL_MAP = json.load(open(LABEL_MAP_PATH))

STRATEGIES = ['random', 'greedy', 'approx_greedy', 'neighbor', 'multi_greedy']


def extract_epoch_and_round(fn):
    fn = fn[:-5]
    parts = fn.split('_')
    return int(parts[2]), (parts[4])


for i in range(1, 8):
    with open(f'experiment_inputs/input_{i}.txt', 'r', encoding='utf-8') as f:
        good_fact, bad_fact = f.readlines()
        good_fact = good_fact.strip().split('-')
        bad_fact = bad_fact.strip().split('-')
    base_exp_name = get_base_exp_name(good_fact, bad_fact, LABEL_MAP)

    for mitigator_strategy in STRATEGIES:
        for disinformer_strategy in STRATEGIES:
            exp_name = f'{base_exp_name}_{mitigator_strategy}_{disinformer_strategy}'
            if os.path.exists(f'results/remove_overlapping_budget/{exp_name}'):
                all_results = []
                all_files = os.listdir(
                    f'results/remove_overlapping_budget/{exp_name}')
                all_files.sort()
                for fn in all_files:
                    if 'results_epoch' in fn:
                        # extract epoch and round num
                        epoch, step = extract_epoch_and_round(fn)
                        print("epoch and step:", epoch, step)
                        if len(all_results) <= epoch:
                            all_results.append({})
                        with open(f'results/remove_overlapping_budget/{exp_name}/{fn}', 'r', encoding='utf-8') as curr_res:
                            all_results[epoch][step] = json.load(curr_res)
                with open(f'results/remove_overlapping_budget/{exp_name}/all_results.json', 'w', encoding='utf-8') as outf:
                    json.dump(all_results, outf)
