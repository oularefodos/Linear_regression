import numpy as np

class Linear_regression(object):
    def __init__(self):
        self.weight = None;
        self.bias = None;
    
    def predict(self, x, y):
        return x.dot(self.weight) + self.bias;

    def compute_cost(self, y_predicted, y_real):
        return np.mean((y_predicted - y_real) ** 2)
    
    def fit(self, x, y, epoch, learning_rate):
        self.weight = 0;
        self.bias = 0;
        self.epoch = epoch;
        self.learning_rate = learning_rate;
        
        for ep in range(self.epoch):
            y_predicted = self.predict(x, y);
            cost = self.compute_cost(y_predicted, y);
            print(cost);