# from data.simu_data.module_simu_data import get_data_label 
from package.classifiers.nearest_neighbor import NearestNeighbor
# import numpy as np
# from package.module_data import load_CIFAR10


# cifar10_dir = 'package/data/cifar-10-batches-py'

# Xtr, Ytr, Xte, Yte = load_CIFAR10(cifar10_dir)

# # As a sanity check, we print out the size of the training and test data.
# print('Training data shape: ', Xtr.shape)
# print('Training labels shape: ', Ytr.shape)
# print('Test data shape: ', Xte.shape)
# print('Test labels shape: ', Yte.shape)

# # flatten out all images to be one-dimensional
# Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
# Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

# nn = NearestNeighbor() # create a Nearest Neighbor classifier class
# nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
# Yte_predict = nn.predict(Xte_rows) # predict labels on the test images
# # and now print the classification accuracy, which is the average number
# # of examples that are correctly predicted (i.e. label matches)
# # print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
# print('accuracy: %f', np.mean(Yte_predict, Yte))

# Xtr, Ytr, Xte, Yte = get_data_label()
nn = NearestNeighbor()
# nn.train(Xtr, Ytr)
# print(nn.predict(Xte))
