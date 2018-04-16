from utils.utils import get_args, conver_numeric_column
from utils.config import process_config
from datas.iris_data import DataLoad

def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    #load datas
    data = DataLoad(config)
    (train_x, train_y), (test_x, test_y) = data.load_data()
    my_feature_columns = conver_numeric_column(train_x)



if __name__ == '__main__':
    main()