import numpy as np
from scipy.optimize import minimize
# np.set_printoptions(threshold=np.inf)

from lrCostFunction import lrCostFunction
from ex2.gradientFunctionReg import gradientFunctionReg


def oneVsAll(X, y, num_labels, Lambda):
    """ trains multiple logistic regression classifiers and returns all
        the classifiers in a matrix all_theta, where the i-th row of all_theta
        corresponds to the classifier for label i
    """

    # Some useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.column_stack((np.ones((m, 1)), X))

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the following code to train num_labels
    #               logistic regression classifiers with regularization
    #               parameter lambda.
    #
    # Hint: theta(:) will return a column vector.
    #
    # Hint: You can use y == c to obtain a vector of 1's and 0's that tell use
    #       whether the ground truth is true/false for this class.
    #
    # Note: For this assignment, we recommend using fmincg to optimize the cost
    #       function. It is okay to use a for-loop (for c = 1:num_labels) to
    #       loop over the different classes.

    # Set Initial theta
    initial_theta = np.zeros((n + 1, 1))

    for c in np.arange(1, num_labels+1):
        res = minimize(lrCostFunction, initial_theta, args=(X, (y == c)*1, Lambda), method=None, options={'maxiter':50}) # (y==c) gives an array with true/false values, by doing (y==c)*1 it converts to array with values 0/1
        all_theta[c-1] = res.x

    # for c in range(0, num_labels):
    #     initial_theta = np.zeros((n + 1, 1))
    #     # print('c--> ', c)
    #     result = minimize(lrCostFunction, initial_theta, (X, (y == c), Lambda), method=None, options={ 'maxiter': 50 })
    #     # print('1--> ', result)
    #     all_theta[c][:] = result.x
    #     all_theta[c][0] = 0.0
    #     # print('2--> ', all_theta[c])

    # print('final--> ', all_theta)

    # alpha = 0.3
    # z = np.dot(X, initial_theta)
    # hx = 1 / (1 + np.exp(-1 * z))

    # cost = (np.dot(np.transpose(X), (hx - y))) / m
    # grad = cost + ((Lambda * initial_theta) / m)

    # for i in range(0, num_labels):
    #     a1 = initial_theta - (alpha * grad)
    #     a1 = np.transpose(a1)
    #     all_theta[i] = a1[0][:]
    #     all_theta[i][0] = initial_theta[0][0]

    # This function will return theta and the cost
    # =========================================================================

    return all_theta
