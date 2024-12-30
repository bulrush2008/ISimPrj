
"""
The only main class in PSP module

- "P-": Preconditioning
  read the vtk file (.vtm, .vtr), converting to structured data;
- "S-": Save/Store
  build the hdf5 file to save matrix data. The data would be directly read by
  both the FNN and GAN model;
- "P-": Postprocessing
  read the matrix data, and then convert it to VTK format, which can be
  displayed by Paraview

@author       @date         @aff          @version
Xia, S        2024.12.27    Simpop.cn     v3.x
"""

from pathlib import Path
import numpy as np
import h5py

from Common.writeVTM import writeVTM
from Common.writeVTR import writeVTR

from Common.readVTR import readVTR
from Common.readVTM import readVTM
from Common.assertFileExist import assertFileExist
from Common.idxList import idxList, numOfCases
from Common.paraInList import paraInList, lenParaIn
from Common.cleanBadSpots import cleanBadSpots

class PSP(object):
#==============================================================================
  def __init__(self, mode:str):
  #----------------------------------------------------------------------------
    if mode not in ["HDF2VTK" ,"VTK2HDF"]:
      raise ValueError("Input Must Be Either 'HDF2VTK' Or 'VTK2HDF'")

    self.mode = mode
    pass

  def HDF2VTK(self):
  #----------------------------------------------------------------------------
    """
    Get data from .h5 file, write them to vtm and vtr files, in order to be
    displayed by Paraview
    """
    numOfBlocks = 8
    dirVTM = Path("./FNN_Case_Test")
    if not dirVTM.exists(): dirVTM.mkdir()
    #if dirVTM.exists(): dirVTM.rmdir() # 删除一个空目录

    fileVTM = dirVTM.joinpath("t01.vtm")

    # write the vtm file and return the path of vtr files
    dirVTR = writeVTM(numOfBlocks, fileVTM)
    #print(dirVTR)

    dirVTR = dirVTM.joinpath(dirVTR)
    print(dirVTR)

    if not dirVTR.exists(): dirVTR.mkdir(parents=True)

    dirHDF = Path("../FNN").joinpath("fnn.h5")
    alive = dirHDF.exists()
    print("HDF File exists? ", alive)

    numCoordsEachBlk = [[2,27,2,52,2,12],
                        [2,27,2,52,2,13],
                        [2,27,2,53,2,12],
                        [2,27,2,53,2,13],
                        [2,28,2,52,2,12],
                        [2,28,2,52,2,13],
                        [2,28,2,53,2,12],
                        [2,28,2,53,2,13]]

    # def a lambda func, used just in this routine
    calcPtsNum = lambda l: (l[1]-l[0]+1) * (l[3]-l[2]+1) * (l[5]-l[4]+1)

    # certain block's left and right edge in the data array
    # data[... numPtsL, ..., numPtsR, ...]
    # so data[numPtsL:numPtsR] is the slice we need
    numPtsL = 0
    numptsR = 0

    for idx in range(numOfBlocks):
      numPtsR = calcPtsNum(numCoordsEachBlk[idx]) + numPtsL

      writeVTR(idx, dirVTR, dirHDF, numPtsL, numptsR)

      numPtsL = numptsR
      pass
    pass

  def VTK2HDF(self):
  #----------------------------------------------------------------------------
    """
    Get data from the vtk files and write them to hdf5 file, which serve as a
    database.
    """
    # Cases dir and name

    # check the case number
    if numOfCases != lenParaIn:
      raise ValueError(f"{numOfCases} must equal to {lenParaIn}")

    # all cases are in the directory:
    caseDir = Path("../FSCases")

    # register all cases name to a list of strings
    caseNames = []  # e.g "C003" or "C115"
    for iCase in range(numOfCases):
      s = "%03d"%idxList[iCase]
      caseNames.append("C"+s)
      #print(caseNames[iCase])
      pass

    # assertain each case's path
    casePaths = []
    for iCase in range(numOfCases):
      path = caseDir.joinpath(caseNames[iCase]) 
      casePaths.append(path)
      #print(casePaths[iCase])
      pass

    # MatrixData's directory, the data are integrated with HDF5 format

    # MatrixData dir and name
    h5Path = Path("../FSCases/FSHDF")
    if not h5Path.exists(): h5Path.mkdir()

    h5File = h5Path.joinpath("MatrixData.h5")

    # open the hdf5 file
    hdf = h5py.File(h5File, 'w')

    # loop over each case
    for iCase in range(numOfCases):
      fileNameVTM = Path("case" + "%d"%idxList[iCase] + "_point.002000.vtm")
      filePathVTM = casePaths[iCase].joinpath(fileNameVTM)
      #print(caseNames[i])

      grpC = hdf.create_group(caseNames[iCase])
      grpC.create_dataset("InParam", data=paraInList[iCase])

      # assertain each vtm file is alive
      alive = assertFileExist(filePathVTM)
      if not alive:
        raise LookupError(f"{filePathVTM} Does Not Exsit.")

      # read the only vtm file in this case
      numOfBlock, filePathVTR = readVTM(filePathVTM, idxList[iCase])

      # For certain case, loop all its vtr files, each of which relates to a block
      for jVTR in range(numOfBlock):
        theVTRFile = casePaths[iCase].joinpath(filePathVTR[jVTR].decode("ASCII"))

        alive = assertFileExist(theVTRFile)
        if not alive:
          raise LookupError(f"{theVTRFile} Does Not Exist.")

        ( fieldP,
          fieldU,
          fieldV,
          fieldW,
          fieldT,
          coordsX,
          coordsY,
          coordsZ,
          gIndexRange ) = readVTR(theVTRFile)

        #if iCase==0: print(gIndexRange)
        cleanBadSpots(field=fieldP, gIndexRange=gIndexRange)
        cleanBadSpots(field=fieldT, gIndexRange=gIndexRange)

        # add field data
        grpC.create_dataset("Block-"+"%02d"%jVTR + "-P", data=fieldP)
        grpC.create_dataset("Block-"+"%02d"%jVTR + "-U", data=fieldU)
        grpC.create_dataset("Block-"+"%02d"%jVTR + "-V", data=fieldV)
        grpC.create_dataset("Block-"+"%02d"%jVTR + "-W", data=fieldW)
        grpC.create_dataset("Block-"+"%02d"%jVTR + "-T", data=fieldT)

        # add coordinates
        grpC.create_dataset("Block-"+"%02d"%jVTR + "-X", data=coordsX)
        grpC.create_dataset("Block-"+"%02d"%jVTR + "-Y", data=coordsY)
        grpC.create_dataset("Block-"+"%02d"%jVTR + "-Z", data=coordsZ)
        pass

      #if iCase==0: print(grpC.keys())
      pass
    #print(hdf.keys())

    # close the matrix data file
    hdf.close()
    pass

if __name__=="__main__":
  prompt = input("Input Your Command, Either 'VTK2HDF' Or 'HDF2VTK':\n")

  psp = PSP( prompt )

  if prompt == 'VTK2HDF':
    print("We are writing HDF5 database from vtk files ...")
    psp.VTK2HDF()
  else:
    print("We are writing VTK files from HDF5 database ...")
    psp.HDF2VTK()