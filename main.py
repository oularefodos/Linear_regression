import matplotlib.pyplot as plt
import numpy as np
from Linear_regression import Linear_regression
import sys

def load_dataset(filename):
    try:
        data = np.loadtxt(filename, delimiter=',', skiprows=1)
        x = data[:, 0];
        y = data[:, -1];
        # print("======= show the dataset =======")
        # plt.scatter(x, y);
        # plt.show();
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
    x, y = load_dataset(filename)

    model = Linear_regression();
    model.fit(x, y, 10, 0.001);
    