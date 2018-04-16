import argparse
import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        metavar='C',
        default='../configs/dnn.json',
        help='配置文件')
    args = parser.parse_args()
    return args

def conver_numeric_column(train_x):
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    return my_feature_columns