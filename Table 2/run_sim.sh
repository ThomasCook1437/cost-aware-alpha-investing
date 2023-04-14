#!/bin/bash
#SBATCH --job-name=wealth                      # Job name
#SBATCH --nodes=1                           # Run on a single node
#SBATCH --ntasks-per-core=1                           # Run on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --mem=8GB                           # Job memory request
#SBATCH --time=96:00:00                      # Time limit hrs:min:sec
#SBATCH --output=./logs/slurm-%A-%a.out         # Standard output and error log
#SBATCH --array=[1-10000]

module load gams

eval "$(conda shell.bash hook)"
conda activate gams38

num=$(($SLURM_ARRAY_TASK_ID + 9999 ))

python single_iter.py --dataset_num=$num 

