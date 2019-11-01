from package.classifiers.nearest_neighbor import NearestNeighbor
from package.module_simu_data import get_data_label

Xtr, Ytr, Xte, Yte = get_data_label()
print(Xtr, Ytr, Xte, Yte)
nn = NearestNeighbor()
nn.train(Xtr, Ytr)
print(nn.predict(Xte))
