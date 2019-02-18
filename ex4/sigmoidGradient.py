import numpy as np
from ex2.sigmoid import sigmoid


def sigmoidGradient(z):
    """computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector. In particular, if z is a vector or matrix, you should return
    the gradient for each element."""

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of the sigmoid function evaluated at
    #               each value of z (z can be a matrix, vector or scalar).

    z = np.exp(z) # element wise exponential of x
    z = 1 + z
    sigma = np.true_divide(1, z)
    g = np.multiply(sigma, (1 - sigma))

    # =============================================================

    return g
