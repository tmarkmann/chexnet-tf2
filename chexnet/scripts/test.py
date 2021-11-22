#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
from chexnet.dataloader.cxr14_dataset import CXR14Dataset
from chexnet.model.chexnet import CheXNet
from chexnet.configs.config import chexnet_config
from datetime import datetime


input_shape = (None,
    chexnet_config['data']['image_height'],
    chexnet_config['data']['image_width'],
    chexnet_config['data']['image_channel'])

dataset = CXR14Dataset(chexnet_config)

#Metrics
metric_f1 = tfa.metrics.F1Score(num_classes=len(chexnet_config["data"]["class_names"]), threshold=chexnet_config["test"]["F1_threshold"])
metric_auc = tf.keras.metrics.AUC(curve='ROC',multi_label=True, num_labels=14, from_logits=False)

#Model
model = CheXNet(chexnet_config).model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[metric_f1, metric_auc],
)

# Tensorboard Callback and config logging
log_dir = 'logs/chexnet/' + datetime.now().strftime("evaluation--%Y-%m-%d--%H.%M")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.load_weights()
model.evaluate(
    dataset.ds_testchexnet_config['test']['weights_path'], 
    batch_size=chexnet_config['test']['batch_size'],
    callbacks=[tensorboard_callback])