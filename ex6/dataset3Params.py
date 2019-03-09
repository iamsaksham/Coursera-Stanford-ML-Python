import numpy as np
# import sklearn.svm
from sklearn import svm


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma. You should complete
    this function to return the optimal C and sigma based on a
    cross-validation set.
    """

    # You need to return the following variables correctly.
    C = 1
    sigma = 0.3

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set.
    #               You can use svmPredict to predict the labels on the cross
    #               validation set. For example,
    #                   predictions = svmPredict(model, Xval)
    #               will return the predictions on the cross validation set.
    #
    #  Note: You can compute the prediction error using
    #        mean(double(predictions ~= yval))
    #

    bestC = 0
    bestSigma = 0
    bestScore = 0
    possibleSet = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for sigmaVal in possibleSet: # loop for sigma
      gammaVal = 1.0 / (2.0 * sigmaVal ** 2)
      for CVal in possibleSet: # loop for C
        clf = svm.SVC(C=CVal, kernel='rbf', tol=1e-3, max_iter=200, gamma=gammaVal)
        model = clf.fit(X, y)
        newScore = model.score(Xval,yval)  # mean(double(predictions ~= yval))
        if newScore > bestScore:
          bestScore = newScore
          bestC = CVal
          bestSigma = sigmaVal

    C = bestC
    sigma = bestSigma

    # =========================================================================

    return C, sigma
