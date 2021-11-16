#!/usr/bin/env python3

import tensorflow as tf
from chexnet.dataloader.cxr14_dataset import CXR14Dataset
from chexnet.configs.config import chexnet_config

chexnet_config["dataset"]["download"] = True
#chexnet_config["dataset"]["data_dir"] = "~/tf_datasets"

dataset = CXR14Dataset(chexnet_config)
dataset.benchmark()