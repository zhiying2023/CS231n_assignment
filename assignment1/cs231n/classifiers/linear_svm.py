import numpy as np
from random import shuffle


# from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    y = y.reshape(-1,1)
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            # 梯度
            if margin > 0:
                loss += margin
                dW[:, j] += X[i, :].T
                dW[:, y[i][0]] += -X[i, :].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss = loss / num_train

    # Add regularization to the loss.
    loss = loss + 0.5 * reg * np.sum(W * W)

    dW = 1 / num_train * dW + reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.
    # Rather that first computing the loss and then computing the derivative,
    # it may be simpler to compute the derivative at the same time that the loss is being computed.
    # As a result you may need to modify some of the code above to compute the gradient.#
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    y=y.reshape(-1,1)
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    # 对不同行不同列依次取值
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    score = X.dot(W)
    correct_class_score = score[range(num_train), y[:, 0]].reshape(-1, 1)
    margin = score - correct_class_score + 1
    margin[range(num_train), y[:, 0]] = 0
    loss = np.sum(np.maximum(0, margin))
    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    margin[margin > 0] = 1
    number_sum = margin.sum(axis=1)
    margin[range(margin.shape[0]), y[:,0]] = -number_sum
    dW = np.dot(X.T, margin)
    dW = dW / num_train + reg * W
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
