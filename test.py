import numpy as np

class LinearRegression:
    def __init__(self):
        self.weight = None
        self.bias = None

    def predict(self, X):
        """
        X: array of shape (m,) for single feature
           or (m, n) for n features
        returns y_pred of shape (m,)
        """
        return X.dot(self.weight) + self.bias

    def compute_cost(self, y_pred, y):
        """
        Mean squared error cost J = 1/(2m) * sum((y_pred - y)^2)
        """
        m = len(y)
        return (1/(2*m)) * np.sum((y_pred - y)**2)

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        """
        X: (m,) or (m, n)
        y: (m,)
        """
        # ensure X is 2D for the weight initialization
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        m, n = X.shape
        self.weight = np.zeros(n)
        self.bias = 0.0

        for epoch in range(1, epochs+1):
            # 1) forward pass
            y_pred = self.predict(X)

            # 2) compute gradients
            error = y_pred - y                   # (m,)
            dw    = (1/m) * X.T.dot(error)       # (n,)
            db    = (1/m) * np.sum(error)        # scalar

            # 3) parameter update
            self.weight -= learning_rate * dw
            self.bias   -= learning_rate * db

            # 4) (optional) monitor cost
            if epoch % (epochs // 10) == 0 or epoch == 1:
                cost = self.compute_cost(y_pred, y)
                print(f"Epoch {epoch:4d}/{epochs} — Cost: {cost:.4f}")

        return self

# ------------------------------------------------------------
# Example usage on a simple synthetic dataset:

if __name__ == "__main__":
    # single-feature
    X = np.linspace(0, 10, 100)
    y = 3.5 * X + 4 + np.random.randn(100) * 2
    
    print(X, y)

    # model = LinearRegression()
    # model.fit(X, y, epochs=200, learning_rate=0.001)

    # print("Learned weight:", model.weight, "bias:", model.bias)
