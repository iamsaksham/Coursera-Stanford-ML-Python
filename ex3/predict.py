import numpy as np

from ex2.sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    """ outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    """

    # Useful values
    m, _ = X.shape
    num_labels, _ = Theta2.shape

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The max function might come in useful. In particular, the max
    #       function can also return the index of the max element, for more
    #       information see 'help max'. If your examples are in rows, then, you
    #       can use max(A, [], 2) to obtain the max for each row.
    #

    X = np.column_stack((np.ones((m, 1)), X))
    XT = np.transpose(X)
    z2 = np.dot(Theta1, XT)
    a2 = sigmoid(z2)

    a2 = np.transpose(a2)
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))
    a2 = np.transpose(a2)

    z3 = np.dot(Theta2, a2)
    a3 = sigmoid(z3)
    a3 = np.transpose(a3)

    p = np.argmax(a3, axis=1)

    # =========================================================================

    return p + 1  # add 1 to offset index of maximum in A row
