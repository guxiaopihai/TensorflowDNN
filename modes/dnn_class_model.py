class DnnClassModel:
    def __init__(self,feature_columns,config):
        self.feature_columns = feature_columns
        self.hidden_units = config.hidden_units
        self.n_classes = config.n_classes

    def

