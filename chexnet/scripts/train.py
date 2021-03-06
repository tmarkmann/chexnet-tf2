#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from chexnet.dataloader.cxr14_dataset import CXR14Dataset
from chexnet.model.chexnet import CheXNet
from chexnet.configs.config import chexnet_config

input_shape = (None,
    chexnet_config['data']['image_height'],
    chexnet_config['data']['image_width'],
    chexnet_config['data']['image_channel'])

dataset = CXR14Dataset(chexnet_config)

# Model Definition
train_base = chexnet_config['train']['train_base']
model = CheXNet(chexnet_config, train_base=train_base).model()

optimizer = tf.keras.optimizers.Adam(chexnet_config["train"]["learn_rate"])
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric_auc = tf.keras.metrics.AUC(curve='ROC',multi_label=True, num_labels=len(chexnet_config["data"]["class_names"]), from_logits=False)
metric_bin_accuracy= tf.keras.metrics.BinaryAccuracy()
metric_accuracy = tf.keras.metrics.Accuracy()
metric_f1 = tfa.metrics.F1Score(num_classes=len(chexnet_config["data"]["class_names"]), threshold=chexnet_config["test"]["F1_threshold"], average='macro')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric_auc, metric_bin_accuracy, metric_accuracy, metric_f1],
)

# Tensorboard Callback and config logging
log_dir = 'logs/chexnet/' + datetime.now().strftime("%Y-%m-%d--%H.%M")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

config_matrix = [[k, str(w)] for k, w in chexnet_config["train"].items()]
file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
  tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

# Checkpoint Callback to only save best checkpoint
checkpoint_filepath = 'checkpoint/chexnet/' + datetime.now().strftime("%Y-%m-%d--%H.%M") + '/cp.ckpt'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor=metric_auc.name,
    mode='max',
    save_best_only=True)

# Early Stopping if loss plateaus
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=chexnet_config['train']['early_stopping_patience'])

# Dynamic Learning Rate
dyn_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=chexnet_config['train']['patience_learning_rate'],
    mode="min",
    min_lr=chexnet_config['train']['min_learning_rate'],
)

# Class weights for training underrepresented classes
class_weights = None
if chexnet_config["train"]["use_class_weights"]:
    class_weights = dataset.class_weights

# Model Training
model.fit(
    dataset.ds_train,
    epochs=chexnet_config["train"]["epochs"],
    validation_data=dataset.ds_val,
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping, dyn_lr]
)

#Model Test
model.load_weights(checkpoint_filepath) #best
result = model.evaluate(
    dataset.ds_test, 
    batch_size=chexnet_config['test']['batch_size'],
    callbacks=[tensorboard_callback])

result = dict(zip(model.metrics_names, result))
print(result)
result_matrix = [[k, str(w)] for k, w in result.items()]
with file_writer.as_default():
  tf.summary.text("evaluation", tf.convert_to_tensor(result_matrix), step=0)