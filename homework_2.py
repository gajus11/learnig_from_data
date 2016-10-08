import math
import numpy as np
from numpy import random
from homework_1 import LinearFunction, createRandomPoints, Perceptron

class Coin:

    def __init__(self, n):
        self.heads = 0
        self.flip_n_times(n)

    def flip_n_times(self, n):
        result_array = random.choice(2, n)
        self.heads = np.count_nonzero(result_array)

def exercise_1():
    #algorithm parameters specified in exercise
    single_coin_flip = 10
    repeats = 100000
    number_of_coins = 1000

    #set averages to 0
    v_1_avg = 0
    v_rand_avg = 0
    v_min_avg = 0

    #simulate coin flips
    for i in range(repeats):
        heads_array = [Coin(10).heads for j in range(number_of_coins)]
        v_1 = heads_array[0] / single_coin_flip
        v_rand = random.choice(heads_array) / single_coin_flip
        v_min = min(heads_array) / single_coin_flip
        v_1_avg += v_1
        v_rand_avg += v_rand
        v_min_avg += v_min

    v_1_avg /= repeats
    v_rand_avg /= repeats
    v_min_avg /= repeats

    print(v_1_avg, v_rand_avg, v_min_avg)

def calculate_linear_regression(number_of_points):
    '''
    Create random linear function f and number_of_points random points.
    Calculate linear regression weights and create linear function g with this weights.
    :param number_of_points:
    :return: (points, f, g)
    '''
    # initialize function
    f = LinearFunction()

    # create random points
    points = createRandomPoints(number_of_points)

    linear_output = f.calculate(points)

    #linear regression
    X = points
    Y = np.matrix(linear_output).T
    weights = np.dot(np.linalg.pinv(X), Y)

    # set linear function with perceptron weights
    g = LinearFunction()
    g.setWeights(weights)

    return (points, f, g)

def exercise_5():
    # number of iterations
    repeats = 1000

    #number of random points to learning
    number_of_points = 100

    Ein = 0

    # calculate data for all repeats
    for index in range(repeats):
        (points, f, g) = calculate_linear_regression(number_of_points)

        linear_output = f.calculate(points)
        linear_regression_output = g.calculate(points)

        # find misclassified points
        misclassified = []
        for i in range(number_of_points):
            if linear_output[i] != linear_regression_output[i]:
                misclassified.append(i)

        Ein += len(misclassified) / number_of_points

    Ein /= repeats
    print('Average Ein: ',Ein)

def exercise_6():
    # number of iterations
    repeats = 1000

    #number of random points to learning
    number_of_points = 100
    number_of_new_points = 1000

    Eout = 0

    # calculate data for all repeats
    for index in range(repeats):
        #Linear regression
        (points, f, g) = calculate_linear_regression(number_of_points)

        # Create 1000 points
        new_points = createRandomPoints(number_of_new_points)

        new_linear_output = f.calculate(new_points)
        new_linear_regression_output = g.calculate(new_points)

        # find misclassified points
        new_misclassified = []
        for i in range(number_of_new_points):
            if new_linear_output[i] != new_linear_regression_output[i]:
                new_misclassified.append(i)

        Eout += len(new_misclassified) / number_of_new_points

    Eout /= repeats
    print('Average Eout: ',Eout)

def exercise_7():
    # number of iterations
    repeats = 1

    number_of_points = 10

    average_iterations = 0

    for index in range(repeats):
        #Linear regression
        (points, f, g) = calculate_linear_regression(number_of_points)

        # initialize function and perceptron
        p = Perceptron()
        p.setWeights(g.weights)

        iteration = 0

        # Find solution. Break if too many iterations
        while iteration < 1000000:
            iteration += 1

            # calculate outputs from perceptron and function
            perceptron_output = p.calculate(points)
            linear_output = f.calculate(points)

            # find misclassified points
            misclassified = []

            for i in range(number_of_points):
                p_out = perceptron_output[i]
                l_out = linear_output[i]
                if perceptron_output[i] != linear_output[i]:
                    misclassified.append(i)

            # break if find solution
            if not misclassified:
                average_iterations += iteration
                break

            rand_misc_index = random.choice(misclassified)
            misc_point = points[rand_misc_index]
            misc_output = linear_output[rand_misc_index]

            # learn perceptron
            p.learn(misc_point, misc_output)

    average_iterations /= repeats
    print('Average iterations: ', average_iterations)

def generate_nonlinear_problem(number_of_points):
    # genarate points and output
    points = ((np.random.rand(number_of_points, 3) * 2) - 1)
    output = np.sign(points[:,1]*points[:,1] + points[:,2] * points[:,2] - 0.6)

    # generate 10% noise for output
    for i in range(int(number_of_points / 10)):
        output[i] *= -1

    return (points, output)

def exercise_8():
    number_of_points = 1000

    repeats = 1000

    Ein = 0

    for index in range(repeats):
        (points, output) = generate_nonlinear_problem(number_of_points)

        #linear regression
        X = points
        Y = np.matrix(output).T
        weights = np.dot(np.linalg.pinv(X), Y)

        linear_regression_output = np.sign(np.dot(points, weights))

        misclassified = 0
        for i in range(number_of_points):
            if output[i] != linear_regression_output[i]:
                misclassified += 1

        Ein += misclassified / number_of_points

    Ein /= repeats
    print('Average Ein: ', Ein)

def exercise_9():
    number_of_points = 1000

    repeats = 10000

    Ein = 0
    avg_weights = np.zeros([6,1])

    for index in range(repeats):
        (points, output) = generate_nonlinear_problem(number_of_points)

        #linear regression
        X = []
        for point in points:
            x1 = point[1]
            x2 = point[2]
            X.append([1, x1, x2, x1*x2, x1*x1, x2*x2])
        X = np.matrix(X)
        Y = np.matrix(output).T
        weights = np.dot(np.linalg.pinv(X), Y)

        linear_regression_output = np.sign(np.dot(X, weights))

        misclassified = 0
        for i in range(number_of_points):
            if output[i] != linear_regression_output[i]:
                misclassified += 1

        Ein += misclassified / number_of_points
        avg_weights += weights

    Ein /= repeats
    avg_weights /= repeats
    print('Average Ein: ', Ein)
    print('Average weights: ', avg_weights)

def exercise_10():
    number_of_points = 1000

    repeats = 1000

    Eout = 0

    for index in range(repeats):
        (points, output) = generate_nonlinear_problem(number_of_points)
        (new_points, new_output) = generate_nonlinear_problem(number_of_points)

        #linear regression
        X = []
        for point in points:
            x1 = point[1]
            x2 = point[2]
            X.append([1, x1, x2, x1*x2, x1*x1, x2*x2])
        X = np.matrix(X)
        Y = np.matrix(output).T
        weights = np.dot(np.linalg.pinv(X), Y)

        #linear regression
        X = []
        for point in new_points:
            x1 = point[1]
            x2 = point[2]
            X.append([1, x1, x2, x1*x2, x1*x1, x2*x2])
        X = np.matrix(X)

        new_linear_regression_output = np.sign(np.dot(X, weights))

        misclassified = 0
        for i in range(number_of_points):
            if new_output[i] != new_linear_regression_output[i]:
                misclassified += 1

        Eout += misclassified / number_of_points

    Eout /= repeats
    print('Average Eout: ', Eout)


if __name__ == '__main__':
    exercise_10()
