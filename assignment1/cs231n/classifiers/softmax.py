import numpy as np
from random import shuffle

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
  N, D = X.shape
  C = W.shape[1]
  for i in range(N):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    norm_constant = -np.max(scores)
    prob = np.exp(correct_class_score + norm_constant)/np.sum(np.exp(scores+norm_constant))
    loss -= np.log(prob)
    for j in range(C):
      if j != y[i]:
        prob_j = np.exp(scores[j] + norm_constant)/np.sum(np.exp(scores+norm_constant))
        dW[:, j] += prob_j*X[i]
      else:
        dW[:, j] -= (1-prob)*X[i]

  dW /= N
  loss /= N
  loss += 0.5*reg*np.sum(W*W)

  dW += reg*W
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
  N, D = X.shape
  scores = X.dot(W)
  # normalization constant
  scores -= np.max(scores)
  exp_scores = np.exp(scores)
  probs = exp_scores/np.sum(exp_scores, axis=1)[:, np.newaxis]
  loss = -np.sum(np.log(probs[np.arange(N), y]))/N + 0.5*reg*np.sum(W*W)

  probs[np.arange(N), y] = -(1-probs[np.arange(N), y])
  dW = X.T.dot(probs)/N + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

