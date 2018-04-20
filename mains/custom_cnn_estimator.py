from utils.utils import get_args
from utils.config import process_config
from models.cnn_class_model_custom import CnnModelCustom
from models.my_cnn_class_model import my_model
from datas.mnist_data import DataLoad
from trains.cnntrain import CnnTrains

def main():
    try:
        config_name = "cnn.json"
        args = get_args(config_name)
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    data = DataLoad(config).load_data()
    classifier = CnnModelCustom(config, my_model).getModel()

    trainer = CnnTrains(classifier, config, data)
    trainer.train()


if __name__ == '__main__':
    main()