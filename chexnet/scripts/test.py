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
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.AUC(curve='ROC',multi_label=True, num_labels=14, from_logits=False)],
)
model.load_weights('/home/tmarkmann/chexnet-tf2/chexnet/weights/chexnet_tf1_weights.h5')

#print(dataset.ds_info)
model.evaluate(dataset.ds_test)