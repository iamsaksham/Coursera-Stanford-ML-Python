import numpy as np


def normalEqn(X, y):
    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """
    theta = 0
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#

# ---------------------- Sample Solution ----------------------
    XT = X.transpose()
    XTX = np.dot(XT, X)
    XTXI = np.linalg.inv(XTX)
    XTXIXT = np.dot(XTXI, XT)
    theta = np.dot(XTXIXT, y)

# -------------------------------------------------------------

    return theta

# ============================================================

