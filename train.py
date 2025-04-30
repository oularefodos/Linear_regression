import matplotlib.pyplot as plt
import numpy as np
from Linear_regression import LinearRegression
import sys

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

    plt.scatter(X, y);

    plt.plot(X, y_pred);

    plt.savefig("data_and_line.jpg");
    