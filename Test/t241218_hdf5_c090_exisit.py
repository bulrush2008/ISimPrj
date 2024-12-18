
import h5py

from pathlib import Path

h5FileDir = Path("../FSCases/FSHDF/MatrixData.h5")

#aLive = h5FileDir.exists()
#print(aLive)

h5 = h5py.File(h5FileDir)
print(len(h5.keys()))
print(h5.keys())