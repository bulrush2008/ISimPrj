
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
Xia, S        2024.12.27    Simpop.cn     v4.x
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
from Common.splitData import splitData

class PSP(object):
#==============================================================================
  def __init__(self, mode:str):
  #----------------------------------------------------------------------------
    if mode not in ["HDF2VTK" ,"VTK2HDF"]:
      raise ValueError("Input Must Be Either 'HDF2VTK' Or 'VTK2HDF'")

    self.mode = mode
    pass

  def act(self):
    mode = self.mode

    if mode == "VTK2HDF":
      self._VTK2HDF()
    elif mode == "HDF2VTK":
      self._HDF2VTK()

  def _HDF2VTK(self):
  #----------------------------------------------------------------------------
    """
    Get data from .h5 file, write them to vtm and vtr files, in order to be
    displayed by Paraview
    """
    numOfBlocks = 8
    dirVTM = Path("./FNN_Case_Test")
    if not dirVTM.exists(): dirVTM.mkdir()

    fileVTM = dirVTM.joinpath("t01.vtm")

    # write the vtm file and return the path of vtr files
    dirVTR = writeVTM(numOfBlocks, fileVTM)

    dirVTR = dirVTM.joinpath(dirVTR)
    print(dirVTR)

    if not dirVTR.exists(): dirVTR.mkdir(parents=True)

    dirHDF = Path("../GAN").joinpath("gan.h5")
    alive = dirHDF.exists()
    print("HDF File exists? ", alive)

    # certain block's left and right edge in the data array
    numCoordsEachBlk = [[2,27,2,52,2,12],
                        [2,27,2,52,2,13],
                        [2,27,2,53,2,12],
                        [2,27,2,53,2,13],
                        [2,28,2,52,2,12],
                        [2,28,2,52,2,13],
                        [2,28,2,53,2,12],
                        [2,28,2,53,2,13]]

    # positions is a dictionary
    positions = splitData(numCoordsEachBlk)

    for idx in range(numOfBlocks):
      vFL = positions["Var"][idx]
      vFR = positions["Var"][idx+1]

      xFL = positions["X"][idx]
      xFR = positions["X"][idx+1]

      yFL = positions["Y"][idx]
      yFR = positions["Y"][idx+1]

      zFL = positions["Z"][idx]
      zFR = positions["Z"][idx+1]

      dataBounds = {"Var":[vFL, vFR], "X":[xFL, xFR], "Y":[yFL, yFR], "Z":[zFL, zFR]}

      writeVTR( idxBlk    = idx,
                dirVTR    = dirVTR,
                dirHDF    = dirHDF,
                dataBounds= dataBounds )
      pass
    pass

  def _VTK2HDF(self):
  #----------------------------------------------------------------------------
    """
    Get data from the vtk files and write them to hdf5 file, which serve as a
    database.
    """
    # check the case number
    if numOfCases != lenParaIn:
      raise ValueError(f"{numOfCases} must equal to {lenParaIn}")

    # Cases dir and name
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
  import sys

  action = sys.argv[1]
  psp = PSP( action )

  if action == "VTK2HDF":
    print("Now We Convert All Cases to MatrixData.")
  elif action == "HDF2VTK":
    print("Now We Convert Predicting Field to VTK data from HDF Format.")
  else:
    raise ValueError("'VTK2HDF' or 'HDF2VTK' Are the Only Two Legal Input.")

  psp.act()