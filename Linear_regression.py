import numpy as np
import json
import os

class LinearRegression:
    """
    Simple linear regression using gradient descent.
    """
    def __init__(self,
                 learning_rate: float = 0.001,
                 epochs: int = 7000,
                 fit_intercept: bool = True,
                 normalize: bool = True):
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

    def fit(self, X: np.ndarray, y: np.ndarray, model_name="linear_regression"):
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
        self.save_model(model_name)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=float)
        return X.dot(self.coef_) + self.intercept_
    
    def save_model(self, model_name):
        parameters = {"weight": self.coef_, "bias": self.intercept_}
        try:
            with open("model.json", "w") as f:
                json.dump(parameters, f, indent=4);
        except Exception as e:
            # Handling any other exception
            print(f"An error occurred: {e}")
    

