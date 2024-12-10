
from pathlib import Path
import h5py

filePathHDF = Path("../PSP/MatrixData.h5")

aLive = filePathHDF.exists()
if aLive:
  print("the HDF5 file EXISTS!")

h5 = h5py.File(filePathHDF, 'r')

numOfVarsAll = 0

for idx in range(8):
  datasetName = "Block-" + "%02d"%idx + "-X"
  coordX = h5["C001"][datasetName][:]
  nx = len(coordX)

  datasetName = "Block-" + "%02d"%idx + "-Y"
  coordY = h5["C001"][datasetName][:]
  ny = len(coordY)

  datasetName = "Block-" + "%02d"%idx + "-Z"
  coordZ = h5["C001"][datasetName][:]
  nz = len(coordZ)

  numOfVars = nx * ny * nz
  print(f"Block{idx}'s dim: {numOfVars}")

  numOfVarsAll += numOfVars

print(f"All Block's dim: {numOfVarsAll}")

h5.close()