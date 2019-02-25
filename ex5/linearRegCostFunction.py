import numpy as np


def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
    # Initialize some useful values
    m = y.size  # number of training examples
    J = 0.0
    grad = 0.0

    # ====================== YOUR CODE HERE ===================================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    #

    hx = np.dot(X, theta)
    squaredError = np.square(hx - y)
    sumSquaredError = np.sum(squaredError)
    J = sumSquaredError / (2 * m)

    # regularization
    thetaSq = np.square(theta[1:])
    reg = (Lambda / (2 * m)) * thetaSq.sum()

    J = J + reg

    # =========================================================================

    return J, grad

