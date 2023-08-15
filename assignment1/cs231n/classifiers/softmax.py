import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_dim, num_class = W.shape
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    score = np.dot(X, W)
    # 解决数值不稳定问题，假设所有的x_i都等于某个常数c，理论上对所有x_i上式结果为1/n。
    # 如果 c 是很小的负数，exp(c)就会下溢，softmax分母会变成0，最后的结果将为NaN
    # 如果 c 量级很大，exp(c)上溢，导致最后的结果将为NaN
    # 减去最大值导致exp(x)中x最大为0，排除了上溢的可能性
    # 同样，分母中至少有一个值为1的项（exp(0)=1），从而也排除了因分母=下溢导致被零除的可能性
    score = score - np.max(score,axis=1).reshape(-1,1)
    score = np.exp(score)
    for i in range(num_train):
        sum_row=np.sum(score[i, :])
        # 损失
        loss = loss - np.log(score[i, y[i]] / sum_row)
        # 梯度
        for j in range(num_class):
            if j != y[i]:
                dW[:, j] = dW[:, j] + score[i,j]*X[i]/sum_row
            else:
                dW[:, j] = dW[:, j] + score[i,j]*X[i]/sum_row-X[i]

    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_dim, num_class = W.shape
    num_train = X.shape[0]
    
    #vectorized gradient
    score = np.dot(X, W)
    score = score - np.max(score,axis=1).reshape(-1,1)
    score = np.exp(score)
    
    #X(N,D)/Y(N,1)可以广播
    ds = score / score.sum(axis=1).reshape(-1,1)
    loss=np.sum(-np.log(ds[range(num_train), y]))

    ds[range(num_train), y] =ds[range(num_train), y]- 1
    dW=np.dot(X.T,ds)
    # ind = np.zeros_like(ds)
    # ind[range(num_train), y] = 1
    # dW = X.T.dot(ds - ind)

    loss = loss / num_train + 0.5 * reg * np.sum(W * W)
    dW = dW / num_train + reg * W
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
