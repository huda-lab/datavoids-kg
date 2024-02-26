from Kelpie.dataset import Dataset
from helpers.helpers import print_fact, extract_subgraph_of_kg, print_sample, find_head_tail_rel
from helpers.constants import SUPPOTED_KG_DATASETS

# arguments user could specify
kg_name = "yago310"

if kg_name not in SUPPOTED_KG_DATASETS:
    raise Exception(f"{kg_name} is not supported!")

base_path = SUPPOTED_KG_DATASETS[kg_name]['base_path']
train_path = base_path + '/train.txt'
test_path = base_path + '/test.txt'
valid_path = base_path + '/valid.txt'

dataset_interface = Dataset(name=kg_name, load=True, train_path=train_path, test_path=test_path, valid_path=valid_path)

print(f'{kg_name} Stats: ')
print("Number of entities", dataset_interface.num_entities)
print("Number of relations", dataset_interface.num_relations)
print("Size of train set", len(dataset_interface.train_triples))

# Sort the output based on the length of training_samples
sorted_output = sorted(dataset_interface.relations, key=lambda rel: len(find_head_tail_rel(dataset_interface, rel_id=dataset_interface.get_id_for_relation_name(rel))), reverse = True)

# Print the sorted output
for index, relation in enumerate(sorted_output):
    try:
        relation_type = dataset_interface.relation_2_type[dataset_interface.get_id_for_relation_name(relation)]
        relation_id = dataset_interface.get_id_for_relation_name(relation)
        training_samples = find_head_tail_rel(dataset_interface, rel_id=relation_id)
        print(relation, relation_type, f'num samples {len(training_samples)}')

    except Exception as e:
        print("failed", relation, e)
