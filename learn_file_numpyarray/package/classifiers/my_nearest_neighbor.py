import numpy as np

class MyNearestNeighbor:
	def __init__(self):
		print('__init__ method')
		pass

	def train(self, train_data, train_label):
		self.train_data = train_data
		self.train_label = train_label

	def predict(self, test_data):
		predict_label = []
		print(type(test_data.shape))
		print(test_data.shape[0])
		num_test = test_data.shape[0]
		for i in range(num_test):
			print(i)
			distances = np.sum(np.abs(self.train_data - test_data[i]), axis=1)
			predict_label.append(self.train_label[np.argmin(distances)])
		return predict_label