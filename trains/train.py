
class DnnTrains:
    def __init__(self, model, config, data):
        self.model = model
        self.batch_size = config.batch_size
        self.train_steps = config.train_steps
        self.data = data


    def train(self):
        self.model.train(input_fn=lambda: self.data.train_input_fn(),steps=self.train_steps)

    def eval(self):
        eval_result = self.model.evaluate(input_fn=lambda: self.data.eval_input_fn())
        return eval_result

    def predict(self, predict_x):
        predict_result = self.model.predict(input_fn=lambda:self.data.eval_input_fn(predict_x))
        return predict_result