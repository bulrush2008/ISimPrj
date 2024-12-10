
from pathlib import Path
import h5py

filePathHDF = Path("../PSP/MatrixData.h5")

aLive = filePathHDF.exists()
if aLive:
  print("the HDF5 file EXISTS!")

h5 = h5py.File(filePathHDF, 'r')

length = len(h5)
print(length, type(length))

h5.close()