#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develgpus
#SBATCH --account=atmlaml
#SBATCH --cpus-per-task=128

source "/p/project/atmlaml/niesel1/venv_soil_lightning/activate.sh"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="/p/project/atmlaml/niesel1/soil-moisture-ML/lightning-hydra-project:$PYTHONPATH"

cd ..
cd src

# train
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" python train.py experiment=soil_experiment model=soil trainer=gpu logger=tensorboard