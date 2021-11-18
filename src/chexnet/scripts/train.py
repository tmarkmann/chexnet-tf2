#!/usr/bin/env python3

import tensorflow as tf
from datetime import datetime
from chexnet.dataloader.cxr14_dataset import CXR14Dataset
from chexnet.model.chexnet import CheXNet
from chexnet.configs.config import chexnet_config

input_shape = (None,
    chexnet_config['data']['image_height'],
    chexnet_config['data']['image_width'],
    chexnet_config['data']['image_channel'])

dataset = CXR14Dataset(chexnet_config)
model = CheXNet(chexnet_config).model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(chexnet_config["train"]["learn_rate"]),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.AUC(curve='ROC',multi_label=True, num_labels=len(chexnet_config["data"]["class_names"]), from_logits=False)],
)

log_dir = 'logs/chexnet/'+ datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(
    dataset.ds_train,
    epochs=chexnet_config["train"]["epochs"],
    validation_data=dataset.ds_test,
    callbacks=[tensorboard_callback]
)