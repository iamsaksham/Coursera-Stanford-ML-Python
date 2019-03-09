import numpy as np


def gaussianKernel(x1, x2, sigma):
    """returns a gaussian kernel between x1 and x2
    and returns the value in sim
    """

    # Ensure that x1 and x2 are column vectors
    #     x1 = x1.ravel()
    #     x2 = x2.ravel()

    # You need to return the following variables correctly.
    sim = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the similarity between x1
    #               and x2 computed using a Gaussian kernel with bandwidth
    #               sigma
    #

    diff = np.square(x1 - x2)
    sum1 = np.sum(diff)
    f1 = sum1 / (2 * sigma * sigma)
    sim = np.exp(-1 * f1)

    # =============================================================
    return sim
