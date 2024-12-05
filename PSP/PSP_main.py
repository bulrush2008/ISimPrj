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

from pathlib import Path

# register all cases names to a list

# case indexes
CommonPath = Path("../FSCases")

numList = [  1,  3,  5,\
            11, 13, 15,\
            21, 23, 25,\
            51, 53, 55,\
            61, 63, 65,\
            71, 73, 75,\
           101,103,105,\
           111,113,115,\
           121,123,125]

# number of cases
NumOfCases = len(numList)

# register each case name to a list of strings
Cases = []
for i in range(NumOfCases):
  s = "%03d"%numList[i]
  Cases.append("C"+s)

# assertain each case's path
CasePath = []
for i in range(NumOfCases):
  path = CommonPath.joinpath(Cases[i]) 
  CasePath.append(path)
  #print(CasePath[i])

for i in range(NumOfCases):
  vtm = "case" + "%d"%numList[i] + "_point.002000.vtm"

  full_vtm = CasePath[i].joinpath(Path(vtm))
  #print(full_vtm)
  #with open()

VTMFileNames = []

