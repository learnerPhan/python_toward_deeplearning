from package.classifiers.nearest_neighbor import NearestNeighbor
from package.module_simu_data import get_data_label

Xtr, Ytr, Xte, Yte = get_data_label()
# Xtr, Ytr, Xte, Yte are of class numpy.ndarray
# Xtr, Ytr are of size (4, 18) and (4,)
# Xte, Yte are of size (2, 18) and (2,)

print(Xtr, Ytr, Xte, Yte)
nn = NearestNeighbor()
nn.train(Xtr, Ytr)
print(nn.predict(Xte))
