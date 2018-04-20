import argparse
import os
import tensorflow as tf


def get_args(config_name):
    config_dir = os.path.join("../configs", config_name)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        metavar='C',
        default=config_dir,
        help='配置文件')
    args = parser.parse_args()
    return args

def conver_numeric_column(train_x):
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    return my_feature_columns