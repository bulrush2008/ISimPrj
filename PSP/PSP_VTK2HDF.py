"""
Get data from the vtk files and write them to hdf5 file, which serve as a
  database.

1, "P-": Preconditioning
  read the vtk file (.vtm, .vtr), converting to structured data;
2, "S-": Save/Store
  build the hdf5 file to save matrix data. The data would be directly read by
  both the FNN and GAN model;
3, "P-": Postprocessing
  read the matrix data, and then convert it to VTK format, which can be
  displayed by Paraview

@author       @date         @aff          @version
Xia, S        2024.11.7     Simpop.cn     v1.0
"""

#------------------------------------------------------------------------------
# Headers
import sys
from pathlib import Path
import numpy as np

from ReadVTM import ReadVTM
from ReadVTR import ReadVTR
import h5py

from Common.AssertFileExist import AssertFileExist
from Common.idxList import idxList, numOfCases
from Common.paraInList import paraInList, lenParaIn  # parameterization inputs

#------------------------------------------------------------------------------
# Cases dir and name

# check the case number
if numOfCases != lenParaIn:
  raise ValueError(f"{numOfCases} must equal to {lenParaIn}")

# all cases are in the directory:
caseDir = Path("../FSCases")

# register all cases name to a list of strings
caseNames = []  # e.g "C003" or "C115"
for i in range(numOfCases):
  s = "%03d"%idxList[i]
  caseNames.append("C"+s)
  #print(caseNames[i])

# assertain each case's path
casePaths = []
for i in range(numOfCases):
  path = caseDir.joinpath(caseNames[i]) 
  casePaths.append(path)
  #print(casePaths[i])

#------------------------------------------------------------------------------
# MatrixData's directory, the data are integrated with HDF5 format

h5Path = Path("../FSCases/FSHDF")
if not h5Path.exists(): h5Path.mkdir()

h5File = h5Path.joinpath("MatrixData.h5")

hdf = h5py.File(h5File, 'w')

# loop over each case
for i in range(numOfCases):
  VTMFileName = Path("case" + "%d"%idxList[i] + "_point.002000.vtm")
  VTMFilePath = casePaths[i].joinpath(VTMFileName)
  #print(caseNames[i])

  grpC = hdf.create_group(caseNames[i])
  grpC.create_dataset("InParam", data=paraInList[i])

  # assertain each vtm file is alive
  alive = AssertFileExist(VTMFilePath)
  if not alive:
    print(VTMFilePath, " Does Not Exsit.")
    sys.exit(1)

  # read the only vtm file in this case
  numOfBlock, VTRFilePath = ReadVTM(VTMFilePath, idxList[i])

  # For certain case, loop all its vtr files, each of which relates to a block
  for j in range(numOfBlock):
    theVTRFile = casePaths[i].joinpath(VTRFilePath[j].decode("ASCII"))

    alive = AssertFileExist(theVTRFile)
    if not alive:
      print(f"{theVTRFile} Does Not Exist.")
      sys.exit(2)

    fieldP, fieldU, fieldV, fieldW, fieldT, \
    coordsX, coordsY, coordsZ = ReadVTR(theVTRFile)

    #if i==0: print(coordsZ)

    # add field data
    grpC.create_dataset("Block-"+"%02d"%j + "-P", data=fieldP)
    grpC.create_dataset("Block-"+"%02d"%j + "-U", data=fieldU)
    grpC.create_dataset("Block-"+"%02d"%j + "-V", data=fieldV)
    grpC.create_dataset("Block-"+"%02d"%j + "-W", data=fieldW)
    grpC.create_dataset("Block-"+"%02d"%j + "-T", data=fieldT)

    # add coordinates
    grpC.create_dataset("Block-"+"%02d"%j + "-X", data=coordsX)
    grpC.create_dataset("Block-"+"%02d"%j + "-Y", data=coordsY)
    grpC.create_dataset("Block-"+"%02d"%j + "-Z", data=coordsZ)

  #if i==0: print(grpC.keys())
#print(hdf.keys())

hdf.close()