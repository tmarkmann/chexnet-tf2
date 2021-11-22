#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
from chexnet.dataloader.cxr14_dataset import CXR14Dataset
from chexnet.model.chexnet import CheXNet
from chexnet.configs.config import chexnet_config

input_shape = (None,
    chexnet_config['data']['image_height'],
    chexnet_config['data']['image_width'],
    chexnet_config['data']['image_channel'])

dataset = CXR14Dataset(chexnet_config)

metric_f1 = tfa.metrics.F1Score(num_classes=len(chexnet_config["data"]["class_names"]))
metric_auc = tf.keras.metrics.AUC(curve='ROC',multi_label=True, num_labels=14, from_logits=False)

model = CheXNet(chexnet_config).model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[metric_f1, metric_auc],
)
model.load_weights()

model.evaluate(
    dataset.ds_test, 
    batch_size=chexnet_config['test']['batch_size'])