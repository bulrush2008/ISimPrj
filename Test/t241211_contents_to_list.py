
from pathlib import Path

import h5py

filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")

aLive = filePathH5.exists()
#print(aLive)

h5 = h5py.File(filePathH5, 'r')

print("len of h5 file: ", len(h5))

print("---- split ----")
import numpy as np
listCase = np.array(list(h5.keys()))
print(listCase)

for C in listCase:
  print(C)