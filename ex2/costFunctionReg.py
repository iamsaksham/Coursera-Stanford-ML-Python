import numpy as np
from costFunction import costFunction


def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta

    JNormal = costFunction(theta, X, y)
    thetaSquare = np.dot(theta, theta) - (theta[0] * theta[0]) # regularisation does not include theta[0] as this was a constant that was already chosen by us.
    J = JNormal + ((Lambda * thetaSquare) / (2 * m))

    # =============================================================

    return J
