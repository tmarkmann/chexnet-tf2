#!/usr/bin/env python3

import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from chexnet.dataloader.kaggleXRay import KaggleXRayDataset
from chexnet.model.chexnet import CheXNet
from chexnet.configs.kaggle_config import kaggle_config
from chexnet.configs.config import chexnet_config
import sys

# set cli arguments
for arg in sys.argv:
    if arg == "--train-base":
        kaggle_config["train"]["train_base"] = True
    elif arg == "--use_chexnet_weights":
        kaggle_config["train"]["use_chexnet_weights"] = True
    elif arg == "--augmentation":
        kaggle_config["train"]["augmentation"] = True

chexnet_config["train"] = kaggle_config["train"]

# Dataset
dataset = KaggleXRayDataset(kaggle_config)

# Model Definition
chexnet = CheXNet(chexnet_config, train_base=chexnet_config['train']['train_base']).model()
if kaggle_config['train']['use_chexnet_weights']:
    chexnet.load_weights("checkpoint/chexnet/best/cp.ckpt")

x = tf.keras.layers.Flatten()(chexnet.layers[-2].output)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=chexnet.layers[0].input, outputs=x)

optimizer = tf.keras.optimizers.Adam(kaggle_config["train"]["learn_rate"])
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metric_auc = tf.keras.metrics.AUC(curve='ROC',multi_label=True, num_labels=len(kaggle_config["data"]["class_names"]), from_logits=False)
metric_bin_accuracy= tf.keras.metrics.BinaryAccuracy()
metric_f1 = tfa.metrics.F1Score(num_classes=len(kaggle_config["data"]["class_names"]), threshold=kaggle_config["test"]["F1_threshold"], average='macro')

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=[metric_auc, metric_bin_accuracy, metric_f1],
)

# Tensorboard Callback and config logging
log_dir = 'logs/kaggle/' + datetime.now().strftime("%Y-%m-%d--%H.%M")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

config_matrix = [[k, str(w)] for k, w in kaggle_config["train"].items()]
file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
  tf.summary.text("config", tf.convert_to_tensor(config_matrix), step=0)

# Checkpoint Callback to only save best checkpoint
checkpoint_dir = 'checkpoint/kaggle/' + datetime.now().strftime("%Y-%m-%d--%H.%M") + '/'
checkpoint_filepath = checkpoint_dir + 'cp.ckpt'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_loss",
    mode='max',
    save_best_only=True)

# Early Stopping if loss plateaus
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=kaggle_config['train']['early_stopping_patience'])

# Dynamic Learning Rate
dyn_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=kaggle_config['train']['patience_learning_rate'],
    mode="min",
    min_lr=kaggle_config['train']['min_learning_rate'],
)

# Model Training
model.fit(
    dataset.ds_train,
    epochs=kaggle_config["train"]["epochs"],
    validation_data=dataset.ds_val,
    callbacks=[tensorboard_callback, checkpoint_callback, early_stopping, dyn_lr]
)

#Model Test
print('##########  Evaluation  ###########')
model.load_weights(checkpoint_filepath) #best
result = model.evaluate(
    dataset.ds_test, 
    batch_size=kaggle_config['test']['batch_size'],
    callbacks=[tensorboard_callback])

result = dict(zip(model.metrics_names, result))
print(result)
result_matrix = [[k, str(w)] for k, w in result.items()]
with file_writer.as_default():
  tf.summary.text("evaluation", tf.convert_to_tensor(result_matrix), step=0)

#Save whole model
model.save(checkpoint_dir + 'model')