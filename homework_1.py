import math
import numpy as np
from numpy import random


class LinearFunction:

    def __init__(self):
        points = (np.random.rand(2,2) * 2) - 1
        a = (points[0,1] - points[1,1]) / (points[0,0] - points[1,0])
        b = points[0,1] - (a * points[0,0])
        self.weights = np.array([b, a, -1]).T

    def setWeights(self, weights):
        self.weights = weights

    def calculate(self, input):
        output = np.dot(input, self.weights)
        return np.sign(output)

    def calculateY(self, x):
        y = -(self.weights[0] + self.weights[1] * x) / self.weights[2]
        return y

class Perceptron:

    def __init__(self):
        self.weights = np.array([0,0,0]).T
        # print('Weights: ', self.weights)

    def setWeights(self, weights):
        self.weights = weights
        # print('Weights: ', self.weights)

    def calculate(self, input):
        output = np.sign(np.dot(input, self.weights))
        return output

    def learn(self, input , output):
        self.weights = self.weights + (input * output)

def createRandomPoints(number_of_points):
    points = ((np.random.rand(number_of_points,3) * 2) - 1)
    points[:, 0] = 1
    return points

if __name__ == "__main__":
    # number of iterations
    repeats = 1000

    #number of random points to learning
    number_of_points = 100

    average_iterations = 0
    average_probability = 0

    probability_calculation_step = 0.01

    # calculate data for all repeats
    for index in range(repeats):
        # initialize function and perceptron
        p = Perceptron()
        f = LinearFunction()

        # create random points
        points = createRandomPoints(number_of_points)

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

                # set linear function with perceptron weights
                g = LinearFunction()
                g.setWeights(p.weights)

                # calculate misclassified area and probability
                area = 0
                for x_100 in range(int(-1 / probability_calculation_step), int(1 / probability_calculation_step)):
                    x = x_100 * probability_calculation_step
                    f_x = f.calculateY(x)
                    if f_x > 2:
                        f_x = 2
                    elif f_x < -2:
                        f_x = -2

                    g_x = g.calculateY(x)
                    if g_x > 2:
                        g_x = 2
                    elif g_x < -2:
                        g_x = -2

                    area += math.fabs(f_x - g_x) * probability_calculation_step

                # add misclassified probability to average
                average_probability += area / 4

                break

            rand_misc_index = random.choice(misclassified)
            misc_point = points[rand_misc_index]
            misc_output = linear_output[rand_misc_index]

            # learn perceptron
            p.learn(misc_point, misc_output)

    average_iterations /= repeats
    average_probability /= repeats
    print('Average iterations: ', average_iterations)
    print('Average probability: ', average_probability)







