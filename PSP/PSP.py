
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
    if str == "HDF2VTK":
      self._HDF2VTK()
    elif str == "VTK2HDF":
      self._VTK2HDF()
    else:
      raise ValueError("Input Must Be Either 'HDF2VTK' Or 'VTK2HDF'")
    pass

  def _HDF2VTK(self):
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

    for idx in range(numOfBlocks):
      writeVTR(idx, dirVTR, dirHDF)
      pass
    pass

  def _VTK2HDF(self):
  #----------------------------------------------------------------------------
    """
    Get data from the vtk files and write them to hdf5 file, which serve as a
    database.
    """
    pass