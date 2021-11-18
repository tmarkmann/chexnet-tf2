#!/usr/bin/env python3

from src.chexnet.dataloader.cxr14_dataset import CXR14Dataset
from src.chexnet.configs.config import chexnet_config

chexnet_config["dataset"]["download"] = True

dataset = CXR14Dataset(chexnet_config)
dataset.benchmark()