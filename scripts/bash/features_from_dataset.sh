#!/bin/bash
##ATHENA
#SBATCH --job-name=multimodal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdnnp2-gpu-a100
#SBATCH --time=00:10:00
#SBATCH --output=data/slurm_logs/slurm-%j.out



eval "$(conda shell.bash hook)"
conda activate clpi_env

python scripts/python/features_from_dataset.py