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


