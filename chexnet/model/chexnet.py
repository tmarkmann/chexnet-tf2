# -*- coding: utf-8 -*-
"""CheXNet model"""

# external
import tensorflow as tf


# TODO
# - class_weights
# - maybe switch to sequential


class CheXNet(tf.keras.Model):
    """CheXNet Model Class"""

    def __init__(self, config, weigths='imagenet'):
        super(CheXNet, self).__init__(name='CheXNet')
        self.config = config
        self.base_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights=weigths,
            pooling=config['model']['pooling'],
            classes=len(config['data']['class_names']),
        )
        self.classifier = tf.keras.layers.Dense(len(config['data']['class_names']),activation="sigmoid", name="predictions")
        self.last_conv_layer="bn"

    def call(self, inputs):
        x = self.base_model(inputs)
        return self.classifier(x)

    def model(self):
        input_shape = (
            self.config['data']['image_height'],
            self.config['data']['image_width'],
            self.config['data']['image_channel']
        )

        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=x, outputs=self.call(x))