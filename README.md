# Datavoids in Knowledge Graphs

This repository contains the latest implementation of the Knowledge Graph querying game presented in our [Datavoids](https://github.com/huda-lab/datavoids) paper, where two agents compete to influence the link prediction ranking of their respective triples following the presence of a simulated data void in the FB15k-237 dataset.

![kg-graphs](/resources/kg.png)

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
where `KG_NAME` should be replaced with one of our supported knowledge graph datasets: `FB15k-237`.

### Step 2: Datavoid curation based on chosen relations

### Step 3: Calculate Preliminary Simulation Statistics

### Step 4: Run Mitigator-vs-Disinformer Simulation

### Step 5: Visualize Simulation Results

## Acknowledgements

We want to thank Rossi et al. for their implementation of the [Kelpie](https://github.com/AndRossi/Kelpie) explainability framework, which we directly use as a Python package (see ```Kelpie_package/```) in our work to power the triple explanations used as a budget an agent utilizes to explain the specific link prediction they are promoting during the simulation.