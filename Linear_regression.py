import numpy as np

class Linear_regression(object):
    def __init__(self):
        self.weight = None
        self.bias = None

    def predict(self, X):
        return X.dot(self.weight) + self.bias

    def compute_cost(self, y_pred, y):
        m = len(y)
        return (1/(2*m)) * np.sum((y_pred - y)**2)

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        m, n = X.shape
        self.weight = np.zeros(n)
        self.bias = 0.0

        for epoch in range(1, epochs+1):
            y_pred = self.predict(X)

            error = y_pred - y                 
            dw    = (1/m) * X.T.dot(error)       
            db    = (1/m) * np.sum(error)        

            self.weight -= learning_rate * dw
            self.bias   -= learning_rate * db

            if epoch % (epochs // 10) == 0 or epoch == 1:
                cost = self.compute_cost(y_pred, y)
                print(f"Epoch {epoch:4d}/{epochs} — Cost: {cost:.4f}")

        return self