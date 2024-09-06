#!/bin/bash

#SBATCH --job-name=explore # job name
#SBATCH --output=explore.out # output log file
#SBATCH --error=explore.err  # error file
#SBATCH --account=pi-adukia # account info
#SBATCH --time=35:00:00  # 30 hours of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=20       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:4     # Request 1 GPU
#SBATCH --mem=40000 # memory in MB


# running grid detection code
module load cuda
module load python/cpython-3.8.5
source /project2/adukia/miie/image_analysis/environments/zero_shot_classification/bin/activate
python3.8 run_zero_shot_classification.py
deactivate
