# -*- coding: utf-8 -*-
"""Model config in json format"""

kaggle_config = {
    "dataset": {
        "download": False,
        "data_dir": "~/tensorflow_datasets",
    },
    "data": {
        "class_names": ["PNEUMONIA"],
        "image_dimension": (224, 224, 3),
        "image_height": 224,
        "image_width": 224,
        "image_channel": 3,
    },
    "train": {
        "train_base": False,
        "use_chexnet_weights": False,
        "augmentation": False,
        "batch_size": 64,
        "learn_rate": 0.001,
        "epochs": 100,
        "patience_learning_rate": 2,
        "min_learning_rate": 1e-8,
        "early_stopping_patience": 8
    },
    "test": {
        "batch_size": 64,
        "F1_threshold": 0.5,
    },
    "model": {
        "pooling": "avg",
    }
}