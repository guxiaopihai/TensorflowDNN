from utils.utils import get_args
from utils.config import process_config
from models.cnn_class_model_custom import CnnModelCustom
from models.my_cnn_class_model import my_model
from datas.mnist_data import MnistData
import tensorflow as tf
from trains.cnntrain import CnnTrains


def main():
    try:
        config_name = "cnn.json"
        args = get_args(config_name)
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    ministData = MnistData(config)
    classifier = CnnModelCustom(config, my_model).getModel()

    trainer = CnnTrains(classifier, config, ministData)
    for _ in range(40):
        trainer.train()
        eval_result = trainer.eval()
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()