
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

from Common.writeVTM import writeVTM
from Common.writeVTR import writeVTR
from Common.splitData import splitData
class PSP(object):
#==============================================================================
  def __init__(self):
  #----------------------------------------------------------------------------
    with open("./PSP_2VTK.json", 'r') as inp:
      data = json.load(inp)
      pass

    self.vtm_path = data["vtkDir"]
    self.hdf_path = data["h5Dir"]
    pass

  def act(self):
    self._HDF2VTK()

  def _HDF2VTK(self):
  #----------------------------------------------------------------------------
    """
    Get data from .h5 files in directory, write them to vtm and vtr files, in order to be
    displayed by Paraview. Each .h5 file will be processed into its own subdirectory.
    """
    numOfBlocks = 8
    
    # Check if hdf_path is a directory containing .h5 files
    hdf_dir = Path(self.hdf_path)
    if not hdf_dir.exists():
        print(f"Error: HDF directory {hdf_dir} does not exist!")
        return
    
    # Find all .h5 files in the directory
    h5_files = list(hdf_dir.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {hdf_dir}")
        return
    
    print(f"Found {len(h5_files)} .h5 files to process")
    
    # Create main VTM directory
    dirVTM = Path(self.vtm_path)
    if not dirVTM.exists(): 
        dirVTM.mkdir(parents=True)
    
    # Process each .h5 file
    for h5_file in h5_files:
        # Extract case name from filename (remove .h5 extension)
        case_name = h5_file.stem
        print(f"Processing case: {case_name}")
        
        # Create subdirectory for this case
        case_dir = dirVTM.joinpath(case_name)
        if not case_dir.exists():
            case_dir.mkdir(parents=True)
        
        # Create VTM file for this case
        fileVTM = case_dir.joinpath("t01.vtm")
        
        # write the vtm file and return the path of vtr files
        dirVTR = writeVTM(numOfBlocks, fileVTM)
        
        dirVTR = case_dir.joinpath(dirVTR)
        print(f"VTR directory: {dirVTR}")
        
        if not dirVTR.exists(): 
            dirVTR.mkdir(parents=True)
        
        # Current HDF file path
        dirHDF = h5_file
        print(f"Processing HDF file: {dirHDF}")

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

        # Process each block for this case
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
        
        print(f"Completed processing case: {case_name}")
    
    print(f"Finished processing all {len(h5_files)} cases")

if __name__=="__main__":

  psp = PSP()

  # 仅打印提示信息
  print("Now We Convert Multiple Predicting Fields to VTK format from HDF data directory")
  print(f"HDF directory: {psp.hdf_path}")
  print(f"VTK output directory: {psp.vtm_path}")

  # 执行动作
  psp.act()