import numpy as np
from simu_data.data_cifar10 import *

def mergeRGB(data):
	data = data.reshape(2, 3, 2, 3)
	data = data.transpose(0, 2, 3, 1)
	data = data.astype('float')
	return data

def convert_data_label_from_batch_to_nparray(index):
	if index < 2:
		batch = train_batch[index]
	else :
		batch = test_batch
	data = mergeRGB(batch['data'])
	label = np.array(batch['labels'])
	return data, label

def collect_data_label_train(num_train):
	all_train_data = []
	all_train_label = []
	for i in range(0,num_train):
		l_train_data, l_train_label = convert_data_label_from_batch_to_nparray(i) 
		all_train_data.append(l_train_data)
		all_train_label.append(l_train_label)
	train_data = np.concatenate(all_train_data)
	train_label = np.concatenate(all_train_label)
	return train_data, train_label

def get_data_label():
	l_train_data, l_train_label = collect_data_label_train(2)
	l_test_data, l_test_label = convert_data_label_from_batch_to_nparray(2)
	l_train_data = l_train_data.reshape(4, 18)
	l_test_data = l_test_data.reshape(2, 18)
	return l_train_data, l_train_label, l_test_data, l_test_label

if __name__ == '__main__':
	print(get_data_label())



