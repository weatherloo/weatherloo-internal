#!/bin/bash
#SBATCH --job-name=optuna
#SBATCH --array=1-108
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=./logs/optuna-%A-%a.out

mkdir -p ./logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lstm-env

python run_all.py --station eric_d_soulis
