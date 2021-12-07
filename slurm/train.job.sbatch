#!/bin/bash
#SBATCH --gres=gpu:gtx:1
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --output=/media/compute/homes/tmarkmann/train_kaggle.out
#SBATCH --mail-user=tmarkmann@techfak.uni-bielefeld.de
#SBATCH --mail-type=END

srun train.job.sh