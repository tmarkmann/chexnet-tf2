#!/usr/bin/env python3

import tensorflow as tf
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

model.summary()

model.fit(
    dataset.ds_train,
    epochs=chexnet_config["train"]["epochs"],
    validation_data=dataset.ds_test,
)