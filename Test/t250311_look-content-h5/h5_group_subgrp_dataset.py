
import pathlib
import h5py

h5f = h5py.File(pathlib.Path("./temp.h5"), mode='w')

grp1 = h5f.create_group("g1")
sgrp1 = grp1.create_group("sg1")

grp2 = h5f.create_group("g2")

h5f.close()