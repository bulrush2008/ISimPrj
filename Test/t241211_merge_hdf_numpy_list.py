
from pathlib import Path

import h5py
import numpy as np

filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")

aLive = filePathH5.exists()
#print(aLive)

h5 = h5py.File(filePathH5, 'r')

fieldX = []

# into "C001"
for i in range(8):
  pBlk = "Block-" + "%02d"%i + "-X"
  #x = np.array(h5["C001"][pBlk])
  x = list(h5["C001"][pBlk][:])

  fieldX += x

print(fieldX)
