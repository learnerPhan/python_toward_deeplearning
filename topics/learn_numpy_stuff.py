import numpy as np 

'''
here i try to understand this piece of code

    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

'''

#create array from list
x = np.array([0, 0, 0])
print('x.shape : ', x.shape)
print(x)

y = np.array([(1, 1, 1), (2, 2, 2)])
print('y.shape : ', y.shape)
print(y)

#arithmetic
z = x - y
print('z.shape : ', z.shape)
print(z)

a = np.abs(z)
print('a.shape : ', a.shape)
print(a)

b = np.sum(a, axis = 0)
print('b.shape : ', b.shape)
print(b)

c = np.sum(a, axis = 1)
print('c.shape : ', c.shape)
print(c)

d = np.argmax(c)
print('d.shape : ', d.shape)
print(d)
