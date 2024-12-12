
import h5py
from pathlib import Path

hdfDir = Path("../FNN/fnn.h5")
print(hdfDir.exists())


hdf = h5py.File(hdfDir, 'r')

#print(hdf.keys())

#print(hdf["FNN_Out"].keys())

#for ds in hdf["FNN_Out"]:
#  print(ds)

"""
X = hdf["FNN_Out"]["Block-00-X"]
print(X); print(X[:])
Y = hdf["FNN_Out"]["Block-00-Y"]
print(Y); print(Y[:])
Z = hdf["FNN_Out"]["Block-00-Z"]
print(Z); print(Z[:])
"""

P = hdf["FNN_Out"]["Block-00-P"]
print(P)
print(P[100])
print(type(P[100]))

