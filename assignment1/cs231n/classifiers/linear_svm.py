#-*- coding: utf-8 -*-
import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights. : 3073 * 10
  - X: A numpy array of shape (N, D) containing a minibatch of data. : 500 * 3073
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means : 500 * 10
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero, (3073,10)

  # compute the loss and the gradient
  num_classes = W.shape[1] # 10
  num_train = X.shape[0] # 500
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W) #(500*3073) * (3073,10) = (500, 10)
    correct_class_score = scores[y[i]] #1*10
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1 , Hinge loss
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i,:] # 
        dW[:,j] += X[i,:] # margin이 양수일 때에만, 대입

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train 
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) #(500,10)

  correct_label_score_idxes = (range(scores.shape[0]), y) #(500,10)
  print(correct_label_score_idxes)

  # length of this vector is N, one correct label score per datapt
  correct_label_scores = scores[correct_label_score_idxes]

  # subtract correct scores (as a column vector) from every cell
  scores_diff = scores - np.reshape(correct_label_scores, (-1, 1))

  # add 1 for the margin.
  scores_diff += 1

  # now zero out all the loss scores for the correct classes.
  scores_diff[correct_label_score_idxes] = 0

  # now zero out all elements less than zero. (b/c of the max() in the hinge)
  indexes_of_neg_nums = np.nonzero(scores_diff < 0)
  scores_diff[indexes_of_neg_nums] = 0

  #now sum over all dimensions
  loss = scores_diff.sum()
  num_train = X.shape[0]
  loss /= num_train
  # add in the regularization part.
  loss += 0.5 * reg * np.sum(W * W)
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
  scores_diff[scores_diff > 0] = 1
  correct_label_vals = scores_diff.sum(axis=1) * -1
  print(shape(scores_diff.sum(axis=1)))
  scores_diff[correct_label_score_idxes] = correct_label_vals

  dW = X.T.dot(scores_diff)
  dW /= num_train
  # add the regularization contribution to the gradient
  dW += reg * W 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
