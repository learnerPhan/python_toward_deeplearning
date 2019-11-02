import numpy as np
from package.classifiers.nearest_neighbor import NearestNeighbor
from package.module_simu_data import get_data_label

Xtr, Ytr, Xte, Yte = get_data_label()
# Xtr, Ytr, Xte, Yte are of class numpy.ndarray
# Xtr, Ytr are of size (4, 18) and (4,)
# Xte, Yte are of size (2, 18) and (2,)

# print(Xtr, Ytr, Xte, Yte)
nn = NearestNeighbor()
nn.train(Xtr, Ytr)
Yte_predict = nn.predict(Xte) 
# print(Yte_predict)
# print(type(Yte_predict == Yte))
# print(Yte_predict == Yte)
# print('accuracy: %f' % ( np.mean(Yte_predict == Yte) ))
print('-----------------')
print('1')
print(Xtr)
print('2')
print(Xte)
print('3')
print(Xte[1])
print('4')
print(Xte[1:])
print('5')
print(Xtr - Xte[1])
print('5')
print(Xtr - Xte[1:])
print(np.abs(Xtr - Xte[1:]))
print(np.sum(np.abs(Xtr - Xte[1:]), axis=0))
print(np.sum(np.abs(Xtr - Xte[1:]), axis=1))
print(np.argmin(np.sum(np.abs(Xtr - Xte[1:]), axis=1)))
