import tensorflow as tf

class MyModel:
    def __init__(self, features, labels, mode, param):
        self.features = features
        self.labels = labels
        self.mode = mode
        self.params = param

    def build_model(self):
        self.create_graph()
        if self.mode != tf.estimator.ModeKeys.PREDICT:
            self.defined_loss_accuracy()

    def defined_loss_accuracy(self):
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
        # Compute evaluation metrics.
        self.accuracy = tf.metrics.accuracy(labels=self.labels,
                                       predictions=self.predicted_classes,
                                       name='acc_op')
        self.metrics = {'accuracy': self.accuracy}
        tf.summary.scalar('accuracy', self.accuracy[1])

    def create_graph(self):
        net = tf.feature_column.input_layer(self.features, self.params['feature_columns'])
        for units in self.params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        self.logits = tf.layers.dense(net, self.params['n_classes'], activation=None)
        self.predicted_classes = tf.argmax(self.logits, 1)

    def predicted_mode(self):
            predictions = {
                'class_ids': self.predicted_classes[:, tf.newaxis],
                'probabilities': tf.nn.softmax(self.logits),
                'logits': self.logits,
            }
            return tf.estimator.EstimatorSpec(self.mode, predictions=predictions)

    def train_mode(self):
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                self.mode, loss=self.loss, train_op=train_op)

    def eval_mode(self):
             return tf.estimator.EstimatorSpec(
                 self.mode, loss=self.loss, eval_metric_ops=self.metrics)


def my_model(features, labels, mode, params):
    model = MyModel(features, labels, mode, params)
    model.build_model()
    if mode == tf.estimator.ModeKeys.TRAIN:
        estimatorspec = model.train_mode()
    elif mode == tf.estimator.ModeKeys.EVAL:
        estimatorspec = model.eval_mode()
    elif mode == tf.estimator.ModeKeys.PREDICT:
        estimatorspec = model.predicted_mode()
    return estimatorspec