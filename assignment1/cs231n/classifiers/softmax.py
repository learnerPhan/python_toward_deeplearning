from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # print(W.shape)
    #(3073, 10)
    # print(X.shape)
    #(500, 3073)

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
        #get the scores of image i
        # print(X[i].shape)
        #(3073,)
        scores_i = X[i].dot(W)
        # print(scores_i.shape)
        #(10,) 

        #stability trick
        scores_i -= np.max(scores_i)
        exp_scores_i = np.exp(scores_i)
        #end stability trick

        sum_ex_i = np.sum(exp_scores_i)

        # add to loss the sum of image i
        loss += np.log(sum_ex_i)
        # do something similar for loss's gradient
        for j in range(num_class):
            dW[:,j] += X[i]*exp_scores_i[j]/sum_ex_i

        #subtract from loss the correct scores of the image i
        loss -= scores_i[y[i]]
        # do something similar for loss's gradient
        dW[:, y[i]] -= X[i]

    # average
    loss /= num_train
    dW /= num_train

    # regu
    loss += reg*np.sum(W*W)
    dW += reg*2*W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    print(W.shape)
    #(3073, 10)
    print(X.shape)
    #(500, 3073)

    num_train = X.shape[0]
    rows = np.arange(num_train)

    scores = X.dot(W)
    print(scores.shape)
    #(500, 10)

    #stability trick
    scores -= np.max(scores, axis=1).reshape(num_train,1)
    #end stability trick

    exp_scores = np.exp(scores)
    sum_exp_scores_i = np.sum(exp_scores, axis=1)
    print(sum_exp_scores_i.shape)
    #(500,)

    correct_exp_scores = exp_scores[rows,y]
    # print(correct_exp_scores.shape)
    #(500,)
    li = correct_exp_scores/sum_exp_scores_i
    # print(li.shape)
    #(500,)
    li = np.log(li)
    loss -= np.sum(li)
    loss /= num_train
    loss += reg*np.sum(W*W)

    # print(exp_scores.shape)
    #(500, 10)
    # print(sum_exp_scores_i.shape)
    #(500,)


    M = exp_scores/sum_exp_scores_i.reshape(num_train,1)
    M[rows, y] -=1
    dW = X.T.dot(M)
    dW /= num_train
    dW += 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
