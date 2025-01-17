
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

from Common.writeVTM import writeVTM
from Common.writeVTR import writeVTR
from Common.splitData import splitData
class PSP(object):
#==============================================================================
  def __init__(self):
  #----------------------------------------------------------------------------
    pass

  def act(self):
    self._HDF2VTK()

  def _HDF2VTK(self):
  #----------------------------------------------------------------------------
    """
    Get data from .h5 file, write them to vtm and vtr files, in order to be
    displayed by Paraview
    """
    numOfBlocks = 8
    dirVTM = Path("./Case_Test")
    if not dirVTM.exists(): dirVTM.mkdir()

    fileVTM = dirVTM.joinpath("t01.vtm")

    # write the vtm file and return the path of vtr files
    dirVTR = writeVTM(numOfBlocks, fileVTM)

    dirVTR = dirVTM.joinpath(dirVTR)
    print(dirVTR)

    if not dirVTR.exists(): dirVTR.mkdir(parents=True)

    #dirHDF = Path("../GAN").joinpath("gan.h5")
    dirHDF = Path("../FNN").joinpath("fnn.h5")
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

if __name__=="__main__":

  psp = PSP()

  # 仅打印提示i信息
  print("Now We Convert Predicting Field to VTK format from HDF data")

  # 执行动作
  psp.act()