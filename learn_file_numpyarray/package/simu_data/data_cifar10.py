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

train_batch[0]['data'] = np.array(train_batch_data)
train_batch[0]['labels'] = np.array([Class0, Class1])

train_batch[1]['data'] = np.array([Image2, Image3])
train_batch[1]['labels'] = np.array([Class2, Class2])

test_batch['data'] = np.array([Image2, Image3])
test_batch['labels'] = np.array([Class2, Class2])
