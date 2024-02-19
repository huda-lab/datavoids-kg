SEED = 42
STATS_COLS = ['fact', 'budget', 'tail degree',  'head entity degree', 'avg degree of budget heads',
              'avg degree of budget tails']

MAX_MIN_NORMALIZATION_COSTS = {
    'FB15k-237': {
        'max_cost': 8642,
        'min_cost': 1
    },
    'yago310': {
        'max_cost': 61044,
        'min_cost': 1
    }
}