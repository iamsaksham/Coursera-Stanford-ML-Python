import numpy as np

from ex2.sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, Lambda):

    """ computes the cost and gradient of the neural network. The
        parameters for the neural network are "unrolled" into the vector
        nn_params and need to be converted back into the weight matrices.

        The returned parameter grad should be a "unrolled" vector of the
        partial derivatives of the neural network.
    """

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    # Obtain Theta1 and Theta2 back from nn_params
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1), order='F').copy()

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, (hidden_layer_size + 1)), order='F').copy()

    # Setup some useful variables
    m, _ = X.shape

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #

    X = np.column_stack((np.ones((m, 1)), X))
    Theta1T = np.transpose(Theta1)
    Theta2T = np.transpose(Theta2)  # 26x10
    JAll = np.zeros((num_labels, 1))

    z2 = np.dot(X, Theta1T) # 5000x25
    a2 = sigmoid(z2)
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))   # 5000x26

    z3 = np.dot(a2, Theta2T)
    hx = sigmoid(z3)    # 5000x10
    hx2 = 1 - hx
    logHX = np.log(hx)
    logHX2 = np.log(hx2)

    for c in np.arange(1, num_labels+1):
        yNew = (y==c)*1    # (y==c) gives an array with true/false values, by doing (y==c)*1 it converts to array with values 0/1
        yNew = np.transpose(yNew)
        Cost = np.dot(yNew, logHX) + np.dot((1 - yNew), logHX2)
        JAll[c-1] = (-1 * Cost[c-1]) / m

    J = JAll.sum()


    # regularization
    sum1 = np.square(Theta1[:,1:])
    sum1 = sum1.sum()
    sum2 = np.square(Theta2[:,1:])
    sum2 = sum2.sum()
    regular = (Lambda / (2 * m)) * (sum1 + sum2)
    J = J + regular

    # backpropagation
    d3 = np.ones((m, num_labels))   # 5000x10
    d2 = np.ones((m, z2.shape[1]))   # 5000x25
    D2 = np.zeros((num_labels, a2.shape[1])) # 10x26
    D1 = np.zeros((z2.shape[1], X.shape[1])) # 25x401
    sigZ2 = sigmoidGradient(z2) # 5000x25
    for c in np.arange(0, m):
        yTemp = np.zeros((num_labels, ))
        yTemp[y[c]-1] = 1 # vector with values for the output layer for ith example
        d3[c, ] = hx[c] - yTemp
        dotProd = np.dot(Theta2T, d3[c, ])
        dotProd = np.delete(dotProd, (0))   # remove the bias row
        d2[c, ] = (dotProd * sigZ2[c, ])

    D2 = D2 + (np.dot(np.transpose(d3), a2))
    D1 = D1 + (np.dot(np.transpose(d2), X))

    Theta1_grad = D1 / m
    Theta2_grad = D2 / m

    # =========================================================================

    # Unroll gradient
    grad = np.hstack((Theta1_grad.T.ravel(), Theta2_grad.T.ravel()))

    return J, grad
