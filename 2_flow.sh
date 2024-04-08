#!/bin/bash

#SBATCH -p nvidia
#SBATCH -t 72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o batch_jobs_out/%j.out

#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH --mail-user=jfg388@nyu.edu

# Load the necessary modules
module load miniconda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jr-env

# Test if PyTorch can access the GPU and mimic the RNG state operation
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Available devices:', torch.cuda.device_count()); print('Current CUDA device:', torch.cuda.current_device()); torch.cuda.set_rng_state(torch.cuda.get_rng_state()); print('RNG state set successfully')"

# Run the main script
python 2_flow.py --kg_name FB15k-237 --rels_to_test /film/actor/film./film/performance/film /film/director/film /tv/tv_producer/programs_produced./tv/tv_producer_term/program --num_heads_to_test 3 --num_attack_budget 25 --overlapping_budget_threshold 10 --diff_rankings 5
