import tensorflow as tf
class DnnClassModel:
    def __init__(self,feature_columns,config):
        self.feature_columns = feature_columns
        self.hidden_units = config.hidden_units
        self.n_classes = config.n_classes

    def getModel(self):
        classifier = tf.estimator.DNNClassifier(
            feature_columns=self.feature_columns,
            hidden_units=self.hidden_units,
            n_classes=self.n_classes)
        return classifier

