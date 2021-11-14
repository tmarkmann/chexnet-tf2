# -*- coding: utf-8 -*-
"""Model config in json format"""

chexnet_config = {
    "data": {
        "class_names": ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule","Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"],
        "image_dimension": (224, 224, 3),
        "image_height": 224,
        "image_width": 224,
        "image_channel": 3,
    },
    "train": {
        "batch_size": 32,
    },
    "test": {
        "batch_size": 32,
    },
    "model": {
        "weigths": "../../data/CheXNet_Keras_0.3.0_weights.h5",
        "pooling": "avg",
    }
}