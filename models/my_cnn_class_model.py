import tensorflow as tf
class MyCnnModel:
    def __init__(self, features, labels, mode, params):
        self.features = features
        self.labels = labels
        self.mode = mode
        self.data_format = params['data_format']
        if self.data_format == 'channels_first':
            self.input_shape = [1, 28, 28]
        else:
            self.input_shape = [28, 28, 1]
        self.image = features
        if isinstance(self.image, dict):
            self.image = features['image']

    def build_model(self):
        self.create_graph()
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.defined_loss_accuracy()
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            self.logits = self.model(self.image, training=True)
        else:
            self.logits = self.model(self.image, training=False)

    def create_graph(self):
        layer = tf.keras.layers
        max_pool = layer.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=self.data_format)
        self.model = tf.keras.Sequential(
            [
                layer.Reshape(self.input_shape),
                layer.Conv2D(32, 5, padding='same', data_format=self.data_format, activation=tf.nn.relu),
                max_pool,
                layer.Conv2D(64, 5, padding='same', data_format=self.data_format, activation=tf.nn.relu),
                max_pool,
                layer.Flatten(),
                layer.Dense(1024, activation=tf.nn.relu),
                layer.Dropout(0.4),
                layer.Dense(10)
            ])


    def train_mode(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            self.mode, loss=self.loss, train_op=train_op)

    def predicted_mode(self):
        predictions = {
            'classes': tf.argmax(self.logits, axis=1),
            'probabilities': tf.nn.softmax(self.logits),
        }
        return tf.estimator.EstimatorSpec(self.mode, predictions=predictions)

    def eval_mode(self):
         return tf.estimator.EstimatorSpec(
             self.mode, loss=self.loss, eval_metric_ops=self.metrics)

    def defined_loss_accuracy(self):
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
        # Compute evaluation metrics.
        self.accuracy = tf.metrics.accuracy(labels=self.labels, redictions=tf.argmax(self.logits, axis=1))
        self.metrics = {'accuracy': self.accuracy}
        tf.summary.scalar('accuracy', self.accuracy[1])

def  my_model(features, labels, mode, params):
    model = MyCnnModel(features, labels, mode, params)
    model.build_model()
    if mode == tf.estimator.ModeKeys.TRAIN:
        estimatorspec = model.train_mode()
    elif mode == tf.estimator.ModeKeys.EVAL:
        estimatorspec = model.eval_mode()
    elif mode == tf.estimator.ModeKeys.PREDICT:
        estimatorspec = model.predicted_mode()
    return estimatorspec