import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    """ computes the cost of using theta as the
    parameter for logistic regression and the
    gradient of the cost w.r.t. to the parameters."""

    # Initialize some useful values
    m = y.size  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Note: grad should have the same dimensions as theta

    z = np.dot(X, theta)
    sigm = sigmoid(z)
    logHx = np.log(sigm)
    logHx1 = np.log(1 + (-1 * sigm))
    J = (np.dot((-1 * y), logHx) - np.dot((1 + (-1 * y)), logHx1)) / m
    # =============================================================

    return J
