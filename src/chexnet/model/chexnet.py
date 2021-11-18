# -*- coding: utf-8 -*-
"""CheXNet model"""

# external
import tensorflow as tf


# TODO
# - class_weights
# - maybe switch to sequential


class CheXNet(tf.keras.Model):
    """CheXNet Model Class"""

    def __init__(self, config, weigths='imagenet', train_base=False):
        super(CheXNet, self).__init__(name='CheXNet')
        self.config = config
        self._input_shape = (
            self.config['data']['image_height'],
            self.config['data']['image_width'],
            self.config['data']['image_channel']
        )
        self.img_input = tf.keras.Input(shape=self._input_shape)
        
        self.base_model = tf.keras.applications.DenseNet121(
            include_top=False,
            input_tensor=self.img_input,
            input_shape=self._input_shape,
            weights=weigths,
            pooling=config['model']['pooling'],
            classes=len(config['data']['class_names']),
        )
        self.base_model.trainable = train_base
        
        self.classifier = tf.keras.layers.Dense(len(config['data']['class_names']),activation="sigmoid", name="predictions")
        self.last_conv_layer="bn"

    def call(self, inputs):
        x = self.base_model(inputs)
        return self.classifier(x)

    def model(self):
        x = self.base_model.output
        predictions = self.classifier(x)
        return tf.keras.Model(inputs=self.img_input, outputs=predictions)
