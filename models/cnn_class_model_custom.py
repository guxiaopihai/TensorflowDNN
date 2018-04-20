import tensorflow as tf

class CnnModelCustom:
    def __init__(self, config, model_fn):
        self.config = config
        self.model_fn = model_fn

    def getModel(self):
        classifier = tf.estimator.Estimator(
            model_fn=self.model_fn,
            model_dir=self.config.model_dir,
            params={
                'data_format': self.config.data_format
            })
        return classifier
