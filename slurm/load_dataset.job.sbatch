#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --output=/media/compute/homes/tmarkmann/load_dataset.out

srun load_dataset.job.sh