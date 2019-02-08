import numpy as np

from ex2.costFunctionReg import costFunctionReg


def lrCostFunction(theta, X, y, Lambda):
    """ computes the cost of using
        theta as the parameter for regularized logistic regression and the
        gradient of the cost w.r.t. to the parameters.
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X * theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations.
    #

    m = y.size  # number of training examples

    z = np.dot(X, theta)
    hx = 1 / (1 + np.exp(-1 * z))
    logHx = np.log(hx)
    logHx1 = np.log(1 + (-1 * hx))
    JNormal = (np.dot((-1 * y), logHx) - np.dot((1 + (-1 * y)), logHx1)) / m

    thetaSquare = np.dot(theta, theta) - (theta[0] * theta[0]) # regularisation does not include theta[0] as this was a constant that was already chosen by us.
    J = JNormal + ((Lambda * thetaSquare) / (2 * m))

    #  =============================================================

    return J
