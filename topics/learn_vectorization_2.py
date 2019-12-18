"""
Here I try to understand this line of code :
dists[i,:] = np.sqrt( np.sum( ( X[i] - self.X_train ) ** 2, axis=1) )

X.shape = (500, 3072)
self.X_train.shape = (5000, 3072)

Docs : https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

"""

import numpy as np

#fake data similar to X and self.X_train but much smaller
test = np.array([[0, 0], [1, 1]])
train = np.array([[1, 1], [2, 2], [3, 3]])

print(test.shape)
print(train.shape)

num_test = test.shape[0]
num_train = train.shape[0]

print('num_test : ', num_test)
print('num_train : ', num_train)

def one_loop() :
	dists = np.zeros((num_test, num_train))
	print(dists)
	for i in range(num_test):
		xi = test[i]
		print('xi.shape : ', xi.shape)
		print('train.shape : ', train.shape)
		#xi has shape = (1,2)
		#train has shape = (3,2)

		sub = xi - train
		print('sub.shape : ', sub.shape) 
		#sub has shape (3,2) --> broadcast happened here

		#OK, so now is simply just matrix operation
		dists[i,:] = np.sum( ( sub ) ** 2, axis=1)
	
	return dists

def no_loop():
	dists = np.zeros((num_test, num_train))

	#original code
	# dists = np.reshape(np.sum(X**2, axis=1), [num_test,1]) + np.sum(self.X_train**2, axis=1) \
	# 														- 2 * np.matmul(X, self.X_train.T)
	su_sq_te = np.sum(test**2, axis=1)
	print('su_sq_te.shape : ', su_sq_te.shape)
	#(2,)

	su_sq_te_rs = np.reshape(su_sq_te, (num_test, 1))
	print('su_sq_te_rs.shape : ', su_sq_te_rs.shape)
	#(2,1)

	su_sq_tr = np.sum(train**2, axis=1)
	print('su_sq_tr.shape : ', su_sq_tr.shape)
	#(3,)

	matmul2 = 2 * np.matmul(test, train.T)
	print('matmul2.shape : ', matmul2.shape)
	#(2,3)

	x = su_sq_te_rs + matmul2
	print('x.shape : ', x.shape)
	#(2,3) --> so here the broadcasting happened

	y = su_sq_tr + matmul2
	print('y.shape : ', y.shape)
	#(2,3) --> so here the broadcasting happened


d = one_loop()
print('d : ', d)

e = no_loop()
