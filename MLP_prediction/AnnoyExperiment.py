import h5py
from annoy import AnnoyIndex
import numpy as np
import random

data = h5py.File("/mnt/home/apricewhelan/projects/gaia-scratch/data/gaiadr3-apogee-bprp-Xy.hdf5","r")

X_ = data['X'][:]
Y_ = data['y'][:]
clean_index = np.where(np.invert(np.isnan(Y_[:,2])))[0]
X_ = X_[clean_index]
Y_ = Y_[clean_index]

X_train = X_[:int(X_.shape[0]*0.8)]
Y_train = Y_[:int(Y_.shape[0]*0.8)]
X_test = X_[int(X_.shape[0]*0.8):]
Y_test = Y_[int(Y_.shape[0]*0.8):]

f = 23  # Length of item vector that will be indexed

t = AnnoyIndex(f, 'angular')

for i in range(X_train.shape[0]):
    t.add_item(i, X_train[i])

t.build(10) # 10 trees
t.save('test.ann')

pred_y = []
for i in range(X_test.shape[0]):
    index = t.get_nns_by_vector(X_test[i], 10)[1:]
    pred_y.append(np.mean(Y_train[index],axis=0))