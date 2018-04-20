import os
import tensorflow as tf
import tempfile
import urllib
import gzip
import shutil
import numpy as np

class DataLoad:
    def __init__(self, config):
        self.config = config

    def maybe_download(self, directory, filename):
        filepath = os.path.join(directory, filename+".gz")
        if tf.gfile.Exists(filepath):
            return filepath
        if not tf.gfile.Exists(directory):
            tf.gfile.MakeDirs(directory)
        url = self.config.down_url + filename + '.gz'
        _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
        print('Downloading %s to %s' % (url, zipped_filepath))
        urllib.request.urlretrieve(url, zipped_filepath)
        with gzip.open(zipped_filepath, 'rb') as f_in, tf.gfile.Open(filepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(zipped_filepath)
        return filepath

    def unzip(self, directory, filename):
        despath = os.path.join(directory, filename)
        orgpath = os.path.join(directory, filename+".gz")
        if tf.gfile.Exists(despath):
            return despath
        if not tf.gfile.Exists(directory):
            tf.gfile.MakeDirs(directory)
        with gzip.open(orgpath, 'rb') as f_in, tf.gfile.Open(despath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        return despath


    def read32(self,bytestream):
        """Read 4 bytes from bytestream as an unsigned 32-bit integer."""

        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    def check_image_file_header(self, filename):
        """Validate that filename corresponds to images for the MNIST dataset."""
        with tf.gfile.Open(filename, 'rb') as bytestream:
            magic = self.read32(bytestream)
            self.read32(bytestream)  # num_images, unused
            rows = self.read32(bytestream)
            cols = self.read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                               bytestream.name))
            if rows != 28 or cols != 28:
                raise ValueError(
                    'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                    (bytestream.name, rows, cols))

    def check_labels_file_header(self, filename):
        """Validate that filename corresponds to labels for the MNIST dataset."""
        with tf.gfile.Open(filename, 'rb') as f:
            magic = self.read32(f)
            self.read32(f)  # num_items, unused
            if magic != 2049:
                raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                               f.name))

    def decode_image(self, image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(self, label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.cast(label, tf.int32)

        label = tf.reshape(label, [])  # label is a scalar
        return label

    def load_data(self):
        train_images_file = self.unzip(self.config.data_dir, self.config.train_images_file)
        train_labels_file = self.unzip(self.config.data_dir, self.config.train_labels_file)
        test_images_file = self.unzip(self.config.data_dir, self.config.test_images_file)
        test_labels_file = self.unzip(self.config.data_dir, self.config.test_labels_file)

        self.check_image_file_header(train_images_file)
        self.check_labels_file_header(train_labels_file)
        self.check_image_file_header(test_images_file)
        self.check_labels_file_header(test_labels_file)


        self.train_images = tf.data.FixedLengthRecordDataset(
            train_images_file, 28 * 28, header_bytes=16).map(self.decode_image)
        self.train_labels = tf.data.FixedLengthRecordDataset(
            train_labels_file, 1, header_bytes=8).map(self.decode_label)
        self.test_images = tf.data.FixedLengthRecordDataset(
            test_images_file, 28 * 28, header_bytes=16).map(self.decode_image)
        self.test_labels = tf.data.FixedLengthRecordDataset(
            test_labels_file, 1, header_bytes=8).map(self.decode_label)
        return self

    def train_input_fn(self):
        dataset = tf.data.Dataset.zip((self.train_images, self.train_labels))
        dataset = dataset.cache().shuffle(buffer_size=50000).repeat().batch(self.config.batch_size)
        return dataset

    def eval_input_fn(self):
        dataset = tf.data.Dataset.zip((self.test_images, self.test_labels))
        dataset = dataset.batch(self.config.batch_size).make_one_shot_iterator().get_next()
        return dataset
