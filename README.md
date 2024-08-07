# Datavoids in Knowledge Graphs

This repository contains the latest implementation of the Knowledge Graph querying game presented in our [Datavoids](https://github.com/huda-lab/datavoids) paper, where two agents compete to influence the link prediction ranking of their respective triples following the presence of a simulated data void in the FB15k-237 dataset.

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

    Make sure that the Kelpie package is installed.

    ```
    cd Kelpie_package/ 
    pip install .
    ```

## Simulation Steps

### Step 1: Knowledge Graph High-Level Analysis

Here, we take a high-level view of the knowledge graph to pick relations that might be interesting to explore further. We considered highly populated, one-to-many or many-to-many relations as a conceptual proxy for misinformation (e.g., Ben Affleck might have *directed* many movies, some being true and others false, but he was *only* born in one place). However, we encourage exploration of other relation types. Some of the things we generate here are: initial statistics on the specified knowledge graph (KG), including number of triples, number of entities, highest degree node, and a display of relations along with their types, allowing users to select specific relations for further analysis.

To run this step, execute the following command:
```
python 1_kg_analysis.py --kg_name FB15k-237
```
where `kg_name` should be replaced with one of our supported knowledge graph datasets: `FB15k-237`.

## Step 2: Data Void Curation Based on Chosen Relations

After selecting the relations of interest in Step 1, we generate a list of candidate pairs for each relation to run the simulation. The process begins by selecting the highest degree head entity nodes for testing. For each head entity and relation pair, we generate combinations with their respective highest degree tail nodes. We then create a 5% reduced dataset for each combination and run tail prediction to obtain the initial ranks. Finally, we calculate the overlap in the Kelpie explanations for each pair. This approach allows us to identify potential data void scenarios and prepare the dataset for subsequent simulation steps.

The results for this step are saved in `results/{kg_name}/{relation_name}/`.

To run this script, execute the following command:

```
python 2_datavoid_curation.py --kg_name FB15k-237 --rels_to_test /film/actor/film./film/performance/film /film/director/film /tv/tv_producer/programs_produced./tv/tv_producer_term/program --num_heads_to_test 3 --num_attack_budget 25 --num_tails_per_head 6 --overlapping_budget_threshold 10 --diff_rankings 5
```
where
* `kg_name`: The dataset name (in our case, `FB15k-237`).
* `rels_to_test`: The relations you want to test, obtained from Step 1, separated by spaces.
* `num_heads_to_test`: The number of head entities to consider for each relation.
* `num_tails_per_head`: The number of tail entities to test for each head entity.
* `num_attack_budget`: The budget each prospective agent candidate in a pair will have.
* `overlapping_budget_threshold`: The allowed budget overlap between the candidates in a given pair. Lower values are better.
* `diff_rankings`: The maximum allowed difference in rankings for given candidate pairs. Lower values are better.

### Step 3: Calculate Preliminary Simulation Statistics

### Step 4: Run Mitigator-vs-Disinformer Simulation

### Step 5: Visualize Simulation Results

## Acknowledgements

We want to thank Rossi et al. for their implementation of the [Kelpie](https://github.com/AndRossi/Kelpie) explainability framework, which we directly use as a Python package (see ```Kelpie_package/```) in our work to power the triple explanations used as a budget an agent utilizes to explain the specific link prediction they are promoting during the simulation.