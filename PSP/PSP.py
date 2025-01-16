
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
Xia, S        2025.1.16     Simpop.cn     v5.x
"""

from pathlib import Path
import numpy as np
import h5py

from Common.writeVTM import writeVTM
from Common.writeVTR import writeVTR

from Common.readVTR import readVTR
from Common.readVTM import readVTM
from Common.assertFileExist import assertFileExist
from Common.caseList import PSP_read_csv
from Common.paramInps import paramInpDict, lenParamInp
from Common.cleanBadSpots import cleanBadSpots
from Common.splitData import splitData

class PSP(object):
#==============================================================================
  def __init__(self, mode:str, inpFile:Path, outDir:Path):
  #----------------------------------------------------------------------------
    if mode not in ["HDF2VTK" ,"VTK2HDF"]:
      raise ValueError("Input Must Be Either 'HDF2VTK' Or 'VTK2HDF'")

    self.mode = mode
    self.inpF = inpFile
    self.outD = outDir
    pass

  def act(self):
    mode = self.mode

    if mode == "VTK2HDF":
      self._VTK2HDF()
    elif mode == "HDF2VTK":
      inpF = self.inpF
      outPath = self.outD
      self._HDF2VTK(inpPath=inpF,outPath=outDir)

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

  def _VTK2HDF(self, inpPath:Path=Path("./PSP.inp"), outPath:Path=Path("../FSCases/FSHDF")):
  #----------------------------------------------------------------------------
    """
    Get data from the vtk files and write them to hdf5 file, which serve as a
    database.
    """
    inpCSV = inpPath

    caseList = PSP_read_csv(inpCSV)

    numOfCases = len(caseList)

    # check the case number
    if numOfCases != lenParamInp:
      raise ValueError(f"{numOfCases} must equal to {lenParamInp}")

    # Cases dir and name
    # all cases are in the directory:
    caseDir = Path("../FSCases")

    # assertain each case's path
    casePaths = []
    for case in caseList:
      path = caseDir.joinpath(case) 
      casePaths.append(path)
      pass

    # MatrixData's directory, the data are integrated with HDF5 format

    # MatrixData dir and name
    h5Path = outPath
    if not h5Path.exists(): h5Path.mkdir()

    h5File = h5Path.joinpath("MatrixData.h5")

    # open the hdf5 file
    hdf = h5py.File(h5File, 'w')

    # loop over each case
    #for iCase in range(numOfCases):
    for iCase, case in enumerate(caseList):
      fileNameVTM = Path("case" + "%d"%(iCase+1) + "_point.002000.vtm")
      filePathVTM = casePaths[iCase].joinpath(fileNameVTM)
      #print(caseNames[i])

      grpC = hdf.create_group(case)
      grpC.create_dataset("InParam", data=paramInpDict[case])

      # assertain each vtm file is alive
      alive = assertFileExist(filePathVTM)
      if not alive:
        raise LookupError(f"{filePathVTM} Does Not Exsit.")

      # read the only vtm file in this case
      numOfBlock, filePathVTR = readVTM(filePathVTM, iCase+1)

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
      pass
    #print(hdf.keys())

    # close the matrix data file
    hdf.close()
    pass

if __name__=="__main__":
  import sys

  action = sys.argv[1]
  inpDir = Path(sys.argv[2])
  outDir = Path(sys.argv[3])

  psp = PSP( mode=action, inpFile=inpDir, outDir=outDir)

  # 仅打印提示i信息
  if action == "VTK2HDF":
    print("Now We Convert All Cases to MatrixData.")
  elif action == "HDF2VTK":
    print("Now We Convert Predicting Field to VTK data from HDF Format.")
  else:
    raise ValueError("'VTK2HDF' or 'HDF2VTK' Are the Only Two Legal Input.")

  # 执行动作
  psp.act()