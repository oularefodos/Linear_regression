import numpy as np
import json
import os

class LinearRegression:
    """
    Simple linear regression using gradient descent.
    """
    def __init__(self,
                 learning_rate: float = 0.01,
                 epochs: int = 1400):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_: float = 0.0
        self.intercept_: float = 0.0
        self.mean_: np.ndarray = None
        self.std_: np.ndarray = None
        self.cost_history: list[float] = []

    def _normalize(self, X: np.ndarray):
        self.mean_, self.std_ = X.mean(), X.std() or 1e-8
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._normalize(X)
        m = len(X)
        for epoch in range(1, self.epochs + 1):
            y_pred = X.dot(self.coef_) + self.intercept_
            error = y_pred - y

            # gradients
            dw = (1/m) * X.T.dot(error)
            db = (1/m) * np.sum(error)

            # update
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

            cost = (1/(2*m)) * np.sum(error**2)
            self.cost_history.append(cost)

            # simple logging
            if epoch == 1 or epoch % (self.epochs // 10) == 0:
                print(f"Epoch {epoch}/{self.epochs}, cost={cost:.4f}")

        self.coef_ = self.coef_ / self.std_
        self.intercept_ = (
            self.intercept_ - (self.coef_ * self.mean_).sum()
        )
        self.save_model()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=float)
        return X.dot(self.coef_) + self.intercept_
    
    def save_model(self):
        parameters = {"weight": self.coef_, "bias": self.intercept_}
        try:
            with open("model.json", "w") as f:
                json.dump(parameters, f, indent=4);
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def load_model(self, model_pathname):
        try:
            with open(model_pathname, "r") as file:
                data = json.load(file);
                if not data["weight"] or not data["bias"]:
                    print(data);
                    print("Corrupted file, weight and bias missing");
                    exit(1);
                self.coef_ = data["weight"]
                self.intercept_ = data["bias"]
        except Exception as e:
            print(f"An error occurred: {e}")
