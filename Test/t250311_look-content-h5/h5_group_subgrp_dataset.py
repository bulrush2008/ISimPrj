
import pathlib
import h5py
import numpy as np

h5f = h5py.File(pathlib.Path("./temp.h5"), mode='w')

grp1 = h5f.create_group("g1")
grp1.create_dataset(name="d1", data=np.array([1,2,3]), dtype=float)
sgrp1 = grp1.create_group("sg1")
sgrp1.create_dataset(name="d12", data=np.array([1,2,3]), dtype=np.float32)

grp2 = h5f.create_group("g2")
grp2.create_dataset(name="d2", data=np.array([1,2]), dtype=np.int32)

h5f.close()