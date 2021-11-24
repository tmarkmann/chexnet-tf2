import tensorflow as tf
import tensorflow_datasets as tfds
from kaggleChestXrayImages import Kagglechestxrayimages

class KaggleXRayDataset():
    """CXR14Dataset for CheXNet"""

    def __init__(self, config):
        self.config = config

        (train, validation, test), info = tfds.load(
            'Kagglechestxrayimages',
            split=['train[:88%]', 'train[88%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            download=config["dataset"]["download"],
            data_dir=config["dataset"]["data_dir"],
            with_info=True,
        )
        self.ds_info = info

        self.ds_train = self._build_train_pipeline(train)
        self.ds_val = self._build_test_pipeline(validation)
        self.ds_test = self._build_test_pipeline(test)

    def _build_train_pipeline(self, ds):
        ds = ds.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(self.ds_info.splits['train'].num_examples)
        ds = ds.batch(self.config['train']['batch_size'])
        if self.config["train"]["augmentation"]:
            ds = ds.map(self.augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def _build_test_pipeline(self, ds):
        ds = ds.map(
            self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.config['test']['batch_size'])
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def preprocess(self, image, label):
        height = self.config['data']['image_height']
        width = self.config['data']['image_width']
        image = tf.image.resize_with_pad(
            image, height, width, 
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False)
        return tf.cast(image, tf.float32) / 255., label
    
    def augment_data(self, image, label):
        image = tf.image.random_flip_left_right(image)
        return image, label

    def benchmark(self):
        tfds.benchmark(self.ds_train, batch_size=self.config['train']['batch_size'])
