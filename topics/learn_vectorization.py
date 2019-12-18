import numpy as np

test = np.array([[0, 0], [1, 1]])
train = np.array([[1, 1], [2, 2], [3, 3]])

print(test.shape)
print(train.shape)

def two_loop(test, train):
	dist = np.zeros((2, 3))
	print(dist)
	for i in range(test.shape[0]):
		print(test[i,:])
		for j in range(train.shape[0]):
			print(train[j,:])
			square_val = np.square(test[i,:] - train[j,:])
			print('Hello')
			print(square_val)
			print(square_val.shape)
			dist[i,j] = np.sum(square_val, axis = 0)
	print(dist)

def one_loop(test, train):
	dist = np.zeros((2, 3))
	for i in range(test.shape[0]):
		mat_val = np.tile(test[i,:], (3,1))
		print(mat_val)
		print(train)
		mat_vec = mat_val.reshape(6)
		train_vec = train.reshape(6)
		print(train_vec)
		print(mat_vec)
		square_val = np.square(mat_vec - train_vec)
		print(square_val)
		square_mat = square_val.reshape(3,2)
		print(square_mat)

		dist[i,:] = np.sum(square_mat, axis=1) 
	print(dist)

def no_loop(test, train):
	print(test)
	mat_test = np.tile(test, (1,3))
	print(mat_test)
	print(mat_test.shape)
	print(train)
	mat_train = train.reshape(1,6)
	print(mat_train)
	mat_tile = np.tile(mat_train, (2,1))
	print(mat_tile)



# two_loop(test, train)

# one_loop(test, train)
no_loop(test, train)