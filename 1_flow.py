import argparse

from helpers.helpers import (find_head_tail_rel, get_data_from_kg_name,
                             initialize_nx_graph)


def main(kg_name):
    dataset_interface = get_data_from_kg_name(kg_name)[0]
    dataset_nx_graph = initialize_nx_graph(dataset_interface)

    print(f'{kg_name} Stats: ')
    print("Number of entities", dataset_interface.num_entities)
    print("Number of relations", dataset_interface.num_relations)
    print("Size of train set", len(dataset_interface.train_triples))
    print("Size of test set", len(dataset_interface.test_triples))
    print("Size of valid set", len(dataset_interface.valid_triples))

    max_degree = 0
    node_with_max_degree = None

    for node in dataset_nx_graph.nodes():
        degree = dataset_nx_graph.degree(node)
        if degree > max_degree:
            max_degree = degree
            node_with_max_degree = node

    print("Entity with max degree", dataset_interface.get_name_for_entity_id(
        node_with_max_degree), "with", max_degree)

    # Calculate and print average degree
    average_degree = sum(dict(dataset_nx_graph.degree()).values()
                         ) / dataset_nx_graph.number_of_nodes()
    print(f"Average Degree: {average_degree:.2f}")

    # Sort the output based on the length of training_samples
    sorted_output = sorted(dataset_interface.relations, key=lambda rel: len(find_head_tail_rel(
        dataset_interface, rel_id=dataset_interface.get_id_for_relation_name(rel))), reverse=True)

    print("Relation", "\t", 'Relation Type (1-1,1-N,N-1, N-N)',
          "\t", 'Num training samples')
    for index, relation in enumerate(sorted_output):
        try:
            relation_type = dataset_interface.relation_2_type[dataset_interface.get_id_for_relation_name(
                relation)]
            relation_id = dataset_interface.get_id_for_relation_name(relation)
            training_samples = find_head_tail_rel(
                dataset_interface, rel_id=relation_id)
            print(relation, '\t', relation_type,
                  '\t', f'{len(training_samples)}')

        except Exception as e:
            print("failed", relation, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process KG name.')
    parser.add_argument('--kg_name', type=str,
                        help='Name of the knowledge graph', required=True)
    args = parser.parse_args()
    main(args.kg_name)
