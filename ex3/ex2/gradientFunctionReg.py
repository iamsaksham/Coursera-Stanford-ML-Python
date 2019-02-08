import numpy as np
from gradientFunction import gradientFunction
from sigmoid import sigmoid


def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of a particular choice of theta.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    z = np.dot(X, theta)
    sigm = sigmoid(z)
    grad = (np.dot((sigm - y), X) + (Lambda * theta)) / m
    grad[0] = ((np.dot(sigm - y, X)) / m)[0]

    # =============================================================

    return grad