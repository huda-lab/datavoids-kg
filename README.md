# Datavoids in Knowledge Graphs



## Installation and Running Instructions

1. Ensure access to a GPU-enabled compute instance with Python and Conda installed. For more information on Conda, visit [https://conda.io/projects/conda/en/latest/index.html](https://conda.io/projects/conda/en/latest/index.html).

2. **Setup Environment**

    To run our scripts, you need to set up a Conda environment using our YAML file. Execute the following commands:

    ```
    conda env create -f datavoids_env.yml
    ```

    ```
    conda activate datavoids_env
    ```
## Supported Flows

### Flow 1: KG High-Level Analysis
- Provide initial statistics on the specified knowledge graph (KG), including:
    - Number of triples, number of entities, highest degree node.
- Display relations along with their types, allowing users to select specific relations for further analysis.

To run this flow, execute the following command:

```
python 1_flow.py --kg_name KG_NAME
```

where `KG_NAME` should be replaced with one of our supported knowledge graph datasets: `FB15237` or `yago310`.

### Flow 2: Datavoid curation based on chosen relations
- Generate a list of candidate datavoid relations based on user input of relations from the KG.


FB15k-237 relations:
```
/film/actor/film./film/performance/film
/film/director/film
/tv/tv_producer/programs_produced./tv/tv_producer_term/program

```

command

```
python 2_flow.py --kg_name FB15k-237 --rels_to_test /film/actor/film./film/performance/film /film/director/film /tv/tv_producer/programs_produced./tv/tv_producer_term/program --num_heads_to_test 3 --num_attack_budget 25 --overlapping_budget_threshold 10 --diff_rankings 5

```

how to run in hpc
```
sbatch 2_flow.sh 
```

YAGO3-10 relations:
```

```



### Missing Flows

[] Data void curation
- using the relations chosen above, run ```get_candidates.sh```
- mdify so that we sabe all of the intermediayte smaller kgs generated and tested. 
- maybe would have to change how to choose the head and tail for a given relation?? 

[] Data for simulation preparation

[] Simulation run

[] Visualization

## Code Organization Explanation:
TODO: explain the contents of each python file. 

```helpers/```

```Kelpie_package/```



## Utilities