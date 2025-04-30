from Linear_regression import LinearRegression
import sys

if __name__== "__main__":
    args = sys.argv
    if (len(args) != 2):
        print("incorrect parameters");
        exit(1);
    
    model_filename = args[1];

    model = LinearRegression();
    
    model.load_model(model_filename);
    
    while True:
        x_input = input("please enter the kilometre: ")
        if x_input == "exit":
            exit(0);
        x = int(x_input);
        y_pred = model.predict(x);
        print(f"The predicted value of {x} is equal {y_pred}");