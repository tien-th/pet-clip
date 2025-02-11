#!/bin/bash
#SBATCH --job-name=tiennh       # Job name
#SBATCH --output=w_result_ct-clip.txt      # Output file
#SBATCH --error=w_error_ct-clip.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=4G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

cd /home/user01/aiotlab/htien/pet-clip/scripts
yes | python3 /home/user01/aiotlab/htien/pet-clip/scripts/run_train.py