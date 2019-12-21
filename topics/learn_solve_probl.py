"""
find where the person is in an array
"""
import numpy as np
X = np.array([0, 1, 3, 6, 7, 8])
who = 6

where = np.flatnonzero(X == who)
print(where)
#3

"""
pick up randomly x elements in an array
"""
X = np.array([0,0,1,1,1,5,5,7,8,3,3,3])
num = 3
cand = np.random.choice(X,3)
print(cand)

"""
*learn data structur

Problem : We have a buch of images. We have to pick randomly images of the same given label
How can we design data to able resolve this problem
"""
#Store images in an array 
#example 10 images 
ima_f = np.random.randint(0, 5, 10)
ima_s = np.random.randint(0, 5, 10)
images = np.concatenate((ima_f, ima_s), axis=None)
print(ima_f)
print(ima_s)
print(images)

#keep track label of image
#example label is same as image
labels = images
print(labels)

#here I fake data a little bit
#make a given label
given_lb = labels[3]
print(given_lb)
#make images different a little bit
images = np.square(images)
print(images)

#ok for the data structure part
#now the algorithm

#find indices of all the labels equals to the given on
ind_lb = np.flatnonzero( labels == given_lb)
print(ind_lb)

#pick randomly some labels
random_lb = np.random.choice(ind_lb, len(ind_lb)-1)
print(random_lb)

#show images correspondent
for i in random_lb:
	print(images[i])
	#we expect images[i]=given_lb**2

"""
Extract sub-array from array
"""
print('Extract sub-array from array')
n=5
org = np.array(range(n))
print(org)
org += 4

n=3
sub_ind = list(range(n))
print(sub_ind)
# [0, 1, 2, .., n]

sub = org[sub_ind]
print(sub)
print(org)

"""
Reshape multi-dimension numpy array
-1 trick
"""
print('Reshape multi-dimension numpy array')

#create randomly 2D
arr_2d = np.random.randint(5, size=(3,4))
print(arr_2d)

#reshape
new_shape = (2,6)
arr_2d = np.reshape(arr_2d,new_shape)
print(arr_2d)

#2D to 1D
two2oneD = np.reshape(arr_2d, -1)
print(two2oneD)

#create randomly 3D
ar_3d = np.random.randint(10, size=(3,2,4))
print(ar_3d)

#to 2D
a = np.reshape(ar_3d, (ar_3d.shape[0], -1))
print(a.shape)
print(a)

b = np.reshape(ar_3d, (ar_3d.shape[1], -1))
print(b.shape)
print(b)

c = np.reshape(ar_3d, (ar_3d.shape[2], -1))
print(c.shape)
print(c)

"""
Compute common elements in 2 array
trick : np.sum(X==Y)
"""
print('trick : np.sum(X==Y)')
X = np.random.randint(5, size=5)
Y = np.random.randint(6, size=5)
print(X)
print(Y)
com_elm = np.sum(X==Y)
print(com_elm)

"""
work with dictionary
"""
ret = {}
k_choices = [1, 2, 3]

for k in k_choices:
	ret[k] = []
	for i in range(3):
		ret[k].append(i*k)

print(ret)

partial_sum = np.array([np.sum(v) for k, v in ret.items()])
print(partial_sum)


