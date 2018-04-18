import tensorflow as tf

class DnnModelCustom:
    def __init__(self, feature_columns, config, model_fn):
        self.config = config
        self.model_fn = model_fn
        self.feature_columns = feature_columns

    def getModel(self):
        classifier = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={
                'feature_columns': self.feature_columns,
                # Two hidden layers of 10 nodes each.
                'hidden_units': self.config.hidden_units,
                # The model must choose between 3 classes.
                'n_classes': self.config.n_classes,
            })
        return classifier
