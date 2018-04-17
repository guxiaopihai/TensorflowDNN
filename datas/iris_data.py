import pandas as pd
import tensorflow as tf

class DataLoad:

    def __init__(self, config):
        self.config = config

    def maybe_download(self):
        train_path = tf.keras.utils.get_file(self.config.TRAIN_DATA_URL.split('/')[-1], self.config.TRAIN_DATA_URL)
        test_path = tf.keras.utils.get_file(self.config.TEST_DATA_URL.split('/')[-1], self.config.TEST_DATA_URL)
        return train_path, test_path

    def load_data(self,y_name='Species'):
        train_path, test_path = self.maybe_download()
        train = pd.read_csv(train_path, names=self.config.CSV_COLUMN_NAMES, header=0)
        self.train_x, self.train_y = train, train.pop(y_name)
        test = pd.read_csv(test_path, names=self.config.CSV_COLUMN_NAMES, header=0)
        self.test_x, self.test_y = test, test.pop(y_name)
        return self

    def train_input_fn(self):
        dataset = tf.data.Dataset.from_tensor_slices((dict(self.train_x), self.train_y))
        dataset = dataset.shuffle(1000).repeat().batch(self.config.batch_size)
        return dataset

    def eval_input_fn(self, predict_x=None):
        if predict_x is None:
            features = dict(self.test_x)
            inputs = (features, self.test_y)
        else:
            inputs = predict_x
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        assert self.config.batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(self.config.batch_size)
        return dataset