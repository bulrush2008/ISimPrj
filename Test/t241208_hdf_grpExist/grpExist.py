
"""
1，判断一个组，在 h5 文件中是否存在；
2，判断一个数据，在一个组中是否存在
"""

import h5py
import numpy as np

hdf = h5py.File("t.h5", 'w')

grp1 = hdf.create_group("g1")
grp2 = hdf.create_group("g2")

print(hdf.keys())

a = False
a = "g1" in hdf
print(a)

data11 = np.arange(5)
grp1.create_dataset("d1", data=data11)

b = "d1" in grp1; print("b: ", b)

hdf.close()