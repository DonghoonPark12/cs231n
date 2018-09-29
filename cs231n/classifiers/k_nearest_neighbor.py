import numpy as np
import math
class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X #(5000,3072)
    self.y_train = y #(5000,)
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):#500   
      for j in range(num_train):#5000
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #dists[i,j]=np.sqrt(np.sum(np.square(self.X_train[j] - X[i])))
        dists[i,j]=np.sqrt(np.sum(np.square(self.X_train[j,:] - X[i,:])))
        #dists[i,j] = np.linalg.norm(self.X_train[j] - X[i])
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dists[i,:]=np.sqrt(np.sum(np.square(self.X_train - X[i,:]),axis=1))
      #print(np.sum(np.square(self.X_train - X[i,:]),axis=0).shape)
      #dist[i,:] = np.linalg.norm(self.X_train - X[i,:], axis=1)
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists 

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #print((X**2).shape) #element wise product : (500, 3072)
    #print(((X**2).sum(axis=1)).shape) # (500,)
    #print((((X**2).sum(axis=1))[:,np.newaxis]).shape) # (500,1) for broadcasting
    #print(((self.X_train**2).sum(axis=1)).shape) #(5000,)
    #print(X.dot(self.X_train.T).shape) #(500,5000)
    dists = np.sqrt( (X**2).sum(axis=1)[:,np.newaxis] + (self.X_train**2).sum(axis=1) \
                    -2 * X.dot(self.X_train.T) )  
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      #cloest_y = np.argsort 
      #min_index = np.argmin(dists[i])
      # test ix랑 train 5000개를 비교한 것 중에서 거리가 가까운 것으로 sort하고
      # index를 매긴다. 그리고 거기에 해당하는self.y_train 값을 취한다.
      # 앞에서k번째를 취한다.  
      # closest_y = np.array(self.y_train[np.argsort(dist[i],axis=0)[:k]])

      #closest_y = np.take(self.y_train, np.argsort(dists[i])) #거리가 가장 가까운순서대로 y값을 정렬(np.argsort, np.take)
      #closest_y = closest_y[:k] #k개를 취한다.
      closest_y_indicies = np.argsort(dists[i,:],axis = 0)[:k]
      closest_y = self.y_train[closest_y_indicies] 
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #y_pred[i] = self.y_train[min_index]
      # 취한 y값 중에서 유니크한 값과 그 갯수를 받는다.
      # 가장 갯수가 많은 것을 반환
      #print(values) #k=5 이지만 y값 4,6 이 반환되었다. 3개, 2개 
      #print("counts" + str(counts))
      #(values, counts) = np.unique(closest_y, return_counts=True)
      #y_pred[i] = values[np.argmax(counts)] # count가 가장 많은 4가 y_pred[0]에 저장된다. 
      y_pred[i] = np.argmax(np.bincount(closest_y))
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred
