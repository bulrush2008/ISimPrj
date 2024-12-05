"""
1, "P-": Preconditioning
  read the vtk file (.vtm, .vtr), converting to structured data;
2, "S-": Save/Store
  build the hdf5 file to save matrix data. The data would be directly read by
  both the FNN and GAN model;
3, "P-": Postprocessing
  read the matrix data, and then convert it to VTK format, which can be
  displayed by Paraview

@author       @date         @aff          @version
Xia, S        2024.11.5     Simpop.cn     v1.0
"""

import pathlib

# Give the .vtm file's path
VTMPath = "../"