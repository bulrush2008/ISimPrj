
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
Xia, S        2025.1.17     Simpop.cn     v5.x
"""

from pathlib import Path
import numpy as np
import h5py
import json

from Common.readVTR import readVTR
from Common.readVTM import readVTM
from Common.assertFileExist import assertFileExist
from Common.paramInps import paramInpDict, lenParamInp
from Common.cleanBadSpots import cleanBadSpots
from Common.splitData import splitData

class PSP(object):
#==============================================================================
  def __init__(self):
  #----------------------------------------------------------------------------
    with open("PSP_2HDF.json", 'r') as inp:
      data = json.load(inp)
      pass

    self.vtk_path = data["vtkDir"]
    self.caseList = data["case"]
    self.hdf_path = data["matrixDataDir"]
    self.hdf_file = data["matrixDataFile"]
    pass

  def act(self):
  #----------------------------------------------------------------------------
    self._VTK2HDF()
    pass

  def _VTK2HDF(self):
  #----------------------------------------------------------------------------
    """
    - Get data from the vtk files and,
    - write them to hdf5 file, serving as a database.
    """
    caseList = self.caseList

    numOfCases = len(caseList)

    # check the case number
    if numOfCases != lenParamInp:
      raise ValueError(f"{numOfCases} must equal to {lenParamInp}")

    # Cases dir and name
    # all cases are in the directory:
    caseDir = Path(self.vtk_path)

    # assertain each case's path
    casePaths = []
    for case in caseList:
      path = caseDir.joinpath(case)
      casePaths.append(path)
      pass

    # MatrixData's directory, the data are integrated with HDF5 format
    # MatrixData dir and name
    h5Path = Path(self.hdf_path)
    if not h5Path.exists(): h5Path.mkdir(parents=True)

    h5File = h5Path.joinpath(self.hdf_file)

    # open the hdf5 file
    hdf = h5py.File(h5File, 'w')

    # loop over each case
    #for iCase in range(numOfCases):
    for iCase, case in enumerate(caseList):
      fileNameVTM = Path("case" + "%d"%(iCase+1) + "_point.002000.vtm")
      filePathVTM = casePaths[iCase].joinpath(fileNameVTM)
      #print(caseNames[i])

      # each group 'grpC' represents a case
      # each case including 3 sub-groups
      #   - input parameters
      #   - coordinates
      #   - fields, each field as a variable
      grpC = hdf.create_group(case)

      # input_parameters as a sub-group
      sub_grp_inp = grpC.create_group("input")
      sub_grp_inp.create_dataset("inp", data=paramInpDict[case])

      # assertain each vtm file is alive
      alive = assertFileExist(filePathVTM)
      if not alive:
        raise LookupError(f"{filePathVTM} Does Not Exsit.")

      # read the only vtm file in this case
      numOfBlock, filePathVTR = readVTM(filePathVTM, iCase+1)

      # For certain case, loop all its vtr files, each of which relates to a block
      for jVTR in range(numOfBlock):
        sub_grp_blk = grpC.create_group(f"block{jVTR:03d}")

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
        sub_grp_blk.create_dataset("P", data=fieldP)
        sub_grp_blk.create_dataset("U", data=fieldU)
        sub_grp_blk.create_dataset("V", data=fieldV)
        sub_grp_blk.create_dataset("W", data=fieldW)
        sub_grp_blk.create_dataset("T", data=fieldT)

        # add coordinates
        sub_grp_blk.create_dataset("X", data=coordsX)
        sub_grp_blk.create_dataset("Y", data=coordsY)
        sub_grp_blk.create_dataset("Z", data=coordsZ)
        pass
      pass

    # close the matrix data file
    hdf.close()
    pass

if __name__=="__main__":
#==============================================================================
  psp = PSP()

  # 仅打印提示信息
  print("Now We Convert All Cases to MatrixData.")

  # 执行动作：数据转换
  psp.act()