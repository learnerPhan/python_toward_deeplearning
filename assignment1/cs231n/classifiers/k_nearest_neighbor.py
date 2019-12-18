from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


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
        self.X_train = X
        self.y_train = y

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
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                # print('X[i,:].shape', X[i,:].shape)
                # print('self.X_train[j].shape', self.X_train[j].shape)
                dists[i][j] = np.sqrt(np.sum(np.square(X[i,:] - self.X_train[j]), axis = 0))
                pass

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        print('dists.shape : ', dists.shape)
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
            """
            The first attempt is very slow, even slower then two_loops version
            """
            """
            m_i = np.tile(X[i,:], (num_train,1))
            m_i_vec = m_i.reshape(5000*3072,1)
            train_vec = self.X_train.reshape(5000*3072,1)
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            sq_vec = np.square(m_i_vec - train_vec)
            sq_mat = sq_vec.reshape(5000, 3072)
            dists[i, :] = np.sqrt(np.sum(sq_mat, axis = 1))
            """
            """
            Second attempt borrows idea from https://github.com/jariasf
            (X-Y)^2 = X^2 + Y^2 -2XY
            """
            su_sq_te_mat = np.repeat(np.sum(np.square(X[i,:]), axis = 0), num_train)
            su_sq_tr = np.sum(np.square(self.X_train), axis = 1)
            dists[i,:] = np.sqrt(su_sq_te_mat + su_sq_tr - 2*np.matmul(X[i,:], self.X_train.T))
            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        su_sq_te = np.sum(np.square(X), axis = 1)
        # print('su_sq_te.shape : ', su_sq_te.shape)
        su_sq_te_mat = np.tile(su_sq_te.reshape(num_test,1), (1, num_train)) 
        # print('su_sq_te_mat.shape : ', su_sq_te_mat.shape)

        su_sq_tr = np.sum(np.square(self.X_train), axis = 1)
        # print('su_sq_tr.shape : ', su_sq_tr.shape)
        su_sq_tr_mat = np.tile(su_sq_tr.reshape(1,num_train), ( num_test, 1)) 
        # print('su_sq_tr_mat.shape : ', su_sq_tr_mat.shape)

        X_train_T = np.transpose(self.X_train)
        # print('X_train_T.shape : ', X_train_T.shape)

        dists = su_sq_te_mat + su_sq_tr_mat - 2*np.matmul(X, X_train_T)
        # print('dists.shape : ', dists.shape)

        # dists = np.tile(su_sq_te, (num_train, 1)) + np.tile(su_sq_tr, (num_test, 1)) - dists

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return np.sqrt(dists)

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
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            order = np.argsort(dists[i,:])
            for j in range(k):
                closest_y.append(self.y_train[order[j]])
            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            (val, counts) = np.unique(closest_y, return_counts=True)
            idx = np.argmax(counts)
            y_pred[i] = val[idx]
            pass

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
