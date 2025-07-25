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

# Pliki źródłowy i docelowy (możesz podmienić ścieżki)
NPZ_PATH="/net/pr2/projects/plgrid/plggdnnp/datasets/MM-IMDb/images.npz"
NPY_PATH="/net/pr2/projects/plgrid/plggdnnp/datasets/MM-IMDb/images.npy"

eval "$(conda shell.bash hook)"
conda activate clpi_env

python extract.py "$NPZ_PATH" "$NPY_PATH"