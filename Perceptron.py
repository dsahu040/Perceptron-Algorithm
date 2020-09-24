import random
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np


def file_read(filename):
    with open(filename, "r") as file:
        a = []
        for i in file:
            a.append([float(x) for x in i.strip().split(' ')])
    return a


learning_rate = 0.001


def training(rows, iterator):
    weights = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

    while True:
        iterator -= 1
        global_error = 0
        for row in rows:
            output = perceptron(row, weights)
            error = row[-1] - output
            weights[0] += error
            for j in range(len(row) - 1):
                weights[j + 1] += learning_rate * error * row[j]
            global_error += error * error
        print("Iteration " + str(iterator) + "\tRoot mean square error = " + str(sqrt(global_error / 40)))
        if global_error == 0 or iterator == 0:
            break
    print("***** Decision boundary *****" + "\t" + str(weights[0]) + " + " + str(weights[1]) + " * x +  " + str(weights[2]) + " * y  = 0")
    return weights


def testing(rows, weights):
    error_count = 0
    for i in rows:
        output = perceptron(i, weights)
        error = i[-1] - output
        if error != 0.0:
            error_count += 1
    error_rate = error_count / 2000
    print("error rate for testing data : " + str(error_rate) + "\n")
    return error_rate


def perceptron(input_data, weights):
    theta = weights[0]
    for i in range(len(input_data) - 1):
        theta += weights[i + 1] * input_data[i]
    return 1.0 if theta >= 0 else 0.0


def plot_data(rows, filename, train, error_rate, x_points, y_points):
    input_x1 = []
    input_y1 = []
    input_x2 = []
    input_y2 = []

    for row in rows:
        if row[2] == 1.0:
            input_x1.append(row[0])
            input_y1.append(row[1])
        else:
            input_x2.append(row[0])
            input_y2.append(row[1])
    x = input_x1
    y = input_y1
    plt.scatter(x, y, color='blue')
    x = input_x2
    y = input_y2
    plt.scatter(x, y, color='red')
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    if train:
        plt.title("plot for training date " + filename)
    else :
        plt.title("plot for testing data after training " + filename +
                  "\nError rate :" + str(error_rate))

    x = np.linspace(x_points[0], x_points[1])
    y = np.linspace(y_points[0], y_points[1])
    plt.plot(x, y)
    plt.show()


def main():
    for i in range(1, 11):
        filename = "set" + str(i)
        data = file_read(filename + ".train")
        # training
        weights = training(data, 100)
        y_points = [0, -(weights[0] / weights[2])]
        x_points = [-(weights[0] / weights[1]), 0]
        plot_data(data, filename, True, 0, x_points, y_points)
        # testing
        data = file_read("set.test")
        error_rate = testing(data, weights)
        plot_data(data, filename, False, str(error_rate), x_points, y_points)


if __name__ == "__main__":
    main()
