
import h5py
import numpy as np

groups = ["train", "test", "model"]

fileListTrain = ["Data01.dat", "Data02.dat", "Data03.dat"]
fileListTest  = ["Data04.dat", "Data05.dat"]
fileListModel = ["Data06.dat"]

with h5py.File("mg.h5", 'w') as hdf:
  i = 0
  for grpname in groups:
    g = hdf.create_group(grpname)
    g.create_dataset()

    