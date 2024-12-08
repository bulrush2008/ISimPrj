"""
Get data from the vtk files and write them to hdf5 file, which serve as a database

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

from ReadVTM import ReadVTM
from ReadVTR import ReadVTR
from AssertFileExist import AssertFileExist
import h5py

from pathlib import Path
import numpy as np
import sys

# register all cases names to a list
# case indexes
from idxList import idxList
# number of cases
numOfCases = len(idxList)

commonPath = Path("../FSCases")

# parameterization inputs consist a list
from paraInList import paraInList
numOfCases = len(paraInList)  # also
#print(numOfCases)

# register each case name to a list of strings
caseNames = []  # e.g "Case003" or "Case115"
for i in range(numOfCases):
  s = "%03d"%idxList[i]
  caseNames.append("C"+s)

# assertain each case's path
casePaths = []
for i in range(numOfCases):
  path = commonPath.joinpath(caseNames[i]) 
  casePaths.append(path)
  #print(casePaths[i])

hdf = h5py.File("MatrixData.h5", 'w')

# loop over each case
for i in range(numOfCases):
  VTMFileName = Path("case" + "%d"%idxList[i] + "_point.002000.vtm")
  VTMFilePath = casePaths[i].joinpath(VTMFileName)
  print(caseNames[i])

  grpC = hdf.create_group(caseNames[i])

  # assertain each vtm file is alive
  alive = AssertFileExist(VTMFilePath)
  if not alive:
    print(VTMFilePath, " Does Not Exsit.")
    sys.exit(1)

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

    grpC.create_dataset("Block-"+"%02d"%j + "P", data=fieldP)
    grpC.create_dataset("Block-"+"%02d"%j + "U", data=fieldU)
    grpC.create_dataset("Block-"+"%02d"%j + "V", data=fieldV)
    grpC.create_dataset("Block-"+"%02d"%j + "W", data=fieldW)
    grpC.create_dataset("Block-"+"%02d"%j + "T", data=fieldT)

print(hdf.keys())

hdf.close()