
import h5py
from pathlib import Path

fileHDF = Path("../FNN/fnn.h5")

if fileHDF.exists():
  print("yes, fnn.h5 found.")

h5 = h5py.File(fileHDF)

for grp in h5:
  print(grp)
  for ds in h5[grp]:
    print(ds)