import matplotlib.pyplot as plt
import numpy as np
from Linear_regression import LinearRegression
import sys
import math

def load_dataset(filename):
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        x = data[:, 0];
        y = data[:, -1];
        return (x, y);
    except Exception as e:
        print("An error occurred:", e)
        exit(1);

if __name__== "__main__":
    args = sys.argv
    if (len(args) != 2):
        print("incorrect parameters");
        exit(1);
    
    filename = args[1];
    X, y = load_dataset(filename);

    model = LinearRegression();
    model.fit(X, y);

    y_pred = model.predict(X);
    
    # Calculate the accurancy
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f} — average prediction error in target units")

    #Calculate the percentage of the variance
    ssr = np.sum((y - y_pred) ** 2);
    y_mean = np.mean(y);
    sst = np.sum((y - y_mean) ** 2);
    r2 = 1 - ssr / sst;
    print(f"Precision (R²): {r2*100:.2f}%");

    # The plot of data and the line
    plt.scatter(X, y);
    plt.plot(X, y_pred);
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.savefig("data_and_line.jpg");

    # The plot of the cost
    plt.figure(figsize=(8, 5))
    cost_x = range(len(model.cost_history));
    plt.plot(cost_x, model.cost_history);
    plt.xlabel("Iteration")
    plt.ylabel("Cost (MSE)")
    plt.grid(True)
    plt.savefig("cost_plot.jpg");
    