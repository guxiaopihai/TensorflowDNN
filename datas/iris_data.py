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
        train_x, train_y = train, train.pop(y_name)
        test = pd.read_csv(test_path, names=self.config.CSV_COLUMN_NAMES, header=0)
        test_x, test_y = test, test.pop(y_name)
        return (train_x, train_y), (test_x, test_y)