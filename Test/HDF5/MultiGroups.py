
import h5py
import numpy as np

groups = ["train", "test", "model"]

fileListTrain = ["Data01.dat", "Data02.dat", "Data03.dat"]
fileListTest  = ["Data04.dat", "Data05.dat"]
fileListModel = ["Data06.dat"]

with h5py.File("mg.h5", 'w') as hdf:
  for id, grpname in enumerate(groups):
    igroup = hdf.create_group(grpname)

    if id==0:
      for id2, file in enumerate(fileListTrain):
        data = np.loadtxt(file, delimiter=',')
        igroup.create_dataset(file, data=data)
    elif id==1:
      for id2, file in enumerate(fileListTest):
        data = np.loadtxt(file, delimiter=',')
        igroup.create_dataset(file, data=data)
    elif id==2:
      for id2, file in enumerate(fileListModel):
        data = np.loadtxt(file, delimiter=',')
        igroup.create_dataset(file, data=data)

    