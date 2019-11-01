import numpy as np

train_batch = [{},{}]
test_batch = {}

Image0 = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2]
Image1 = [3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5]
Image2 = [6,6,6,6,6,6,7,7,7,7,7,7,8,8,8,8,8,8]
Image3 = [9,9,9,9,9,9,10,10,10,10,10,10,11,11,11,11,11,11]

Class0 = 0
Class1 = 1
Class2 = 2
Class3 = 3

train_batch_data = []
train_batch_data.append(Image0)
train_batch_data.append(Image1)

train_batch[0]['data'] = np.array([Image0, Image1])
train_batch[0]['labels'] = [Class0, Class1]

train_batch[1]['data'] = np.array([Image2, Image3])
train_batch[1]['labels'] = [Class2, Class2]

test_batch['data'] = np.array([Image2, Image3])
test_batch['labels'] = [Class2, Class2]

#RQ1 : 2 training batches, 1 testing batch
if 0:
	print(len(train_batch))
	print(len(test_batch))

#RQ2 : each batch has 2 images and 2 corresponding classes
#RQ3 : each batch has 2*18 for 2 images of size 18 = 2rows * 3columns * 3channelsRGB
#RQ4 : each data batch is of dictionary structure, having key = {'data', 'labels'}
	   #value of key 'data' is a numpy array
	   #value of key 'labels' is a list
if 0:	
	print(type(train_batch[0]['data']))
	print(type(train_batch[0]['labels']))
	
	print(type(train_batch[1]['data']))
	print(type(train_batch[1]['labels']))
	
	print(type(test_batch['data']))
	print(type(test_batch['labels']))


	print(train_batch[0]['data'].shape)
	print(len(train_batch[0]['labels']))
	
	print(train_batch[1]['data'].shape)
	print(len(train_batch[1]['labels']))
	
	print(test_batch['data'].shape)
	print(len(test_batch['labels']))
