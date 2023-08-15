import numpy as np
import os
import pickle


# from past.builtins import xrange


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
                # not use a l\]oop over dimension.                                    #
                #####################################################################
                dists[i, j] = np.sqrt(np.power(X[i, :] - self.X_train[j, :], 2).sum())
        # print('two', dists)
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
            dists[i, :] = np.sqrt(np.power(X[i, :] - self.X_train, 2).sum(axis=1))
            # print('one', dists,'\n')
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
        X_square = np.square(X).sum(axis=1).reshape(-1, 1)
        train_square = np.square(self.X_train).sum(axis=1).reshape(1, -1)
        dists = np.sqrt(X_square + train_square - 2 * X.dot(self.X_train.T))
        # print('dists', '\n', dists)
        pass
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
        num_test, num_train = dists.shape
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
            # 排序，取前k个
            dist = dists[i, :].argsort()
            dist = self.y_train[dist]
            for j in range(k):
                closest_y.append(dist[j])

            # print('dists\t\t',dists[i,:])
            # print('label\t\t',self.y_train.reshape(1,-1))
            # print('closest_y','\t', closest_y)

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # 分析更适合哪个标签
            counts = {}
            for y in closest_y:
                counts[y] = counts.get(y, 0) + 1
            items = list(counts.items())
            items.sort(key=lambda x: x[1], reverse=True)
            y_pred[i] = items[0][0]
            #########################################################################
            #                           END OF YOUR CODE                            #
            #########################################################################

        return y_pred


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# if __name__ == '__main__':
#     k1=9999;k2=99
#     CIFAR_dir = r'C:\Users\change\Desktop\Visual_Analysis\Data\cifar-10-batches-py'
#     train = unpickle(os.path.join(CIFAR_dir, 'data_batch_1'))
#     traindata = train[b'data'][0:k1,:1024]
#     trainlabel = np.array(train[b'labels'][0:k1]).reshape(-1,1)
#     test = unpickle(os.path.join(CIFAR_dir, 'test_batch'))
#     testdata = test[b'data'][0:k2,:1024]
#     testlabel = np.array(test[b'labels'][0:k2]).reshape(-1,1)


#     method = KNearestNeighbor()
#     # testdata = np.random.randint(1, 10, (10, 3))
#     # traindata = np.random.randint(1, 10, (5, 3))
#     # labeldata = np.array([1, 2, 3, 2, 1]).reshape(5, 1)
#     method.train(traindata, trainlabel)

#     # method.compute_distances_two_loops(testdata)
#     # method.compute_distances_one_loop(testdata)
#     # method.compute_distances_no_loops(testdata)
#     for j in range(1,10):
#         test_pred = method.predict(testdata, k=j)
#         error = 0
#         for i in range(k2):
#             if test_pred[i] != testlabel[i]:
#                 error = error + 1
#         print('错误率：', error / k2)
