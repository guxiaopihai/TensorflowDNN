from utils.utils import get_args, conver_numeric_column
from utils.config import process_config
from datas.iris_data import DataLoad
from models.dnn_class_model_premade import DnnClassModel
from trains.dnntrain import DnnTrains

def main():
    try:
        config_name = "dnn.json"
        args = get_args(config_name)
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    data = DataLoad(config).load_data()
    my_feature_columns = conver_numeric_column(data.train_x)

    classifier = DnnClassModel(my_feature_columns, config).getModel()

    trainer = DnnTrains(classifier, config, data)
    trainer.train()
    eval_result = trainer.eval()
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))



    predictions = trainer.predict(config.predict_x)
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    for pred_dict, expec in zip(predictions, config.SPECIES):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(config.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    main()