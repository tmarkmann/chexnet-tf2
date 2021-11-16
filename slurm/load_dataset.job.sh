#!/bin/bash
source /media/compute/homes/tmarkmann/miniconda3/etc/profile.d/conda.sh
conda activate chexnet
export TFDS_DATA_DIR=/media/compute/homes/tmarkmann/tf_datasets

python3 -m chexnet.scripts.load_dataset