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

idxList = [  1,  3,  5,\
            11, 13, 15,\
            21, 23, 25,\
            51, 53, 55,\
            61, 63, 65,\
            71, 73, 75,\
           101,103,105,\
           111,113,115,\
           121,123,125]

# parameterization inputs: [inlet temperature, inlet velocity, heat flux]
paraInList = [[438.15, 2.511048614, 458333.3333],\
              [458.15, 2.511048614, 458333.3333],\
              [478.15, 2.511048614, 458333.3333],\
              [438.15, 2.064639971, 458333.3333],\
              [458.15, 2.064639971, 458333.3333],\
              [478.15, 2.064639971, 458333.3333],\
              [438.15, 1.618231329, 458333.3333],\
              [458.15, 1.618231329, 458333.3333],\
              [478.15, 1.618231329, 458333.3333],\
              [438.15, 2.511048614, 291666.6667],\
              [458.15, 2.511048614, 291666.6667],\
              [478.15, 2.511048614, 291666.6667],\
              [438.15, 2.064639971, 291666.6667],\
              [458.15, 2.064639971, 291666.6667],\
              [478.15, 2.064639971, 291666.6667],\
              [438.15, 1.618231329, 291666.6667],\
              [458.15, 1.618231329, 291666.6667],\
              [478.15, 1.618231329, 291666.6667],\
              [438.15, 2.511048614, 125000.],\
              [458.15, 2.511048614, 125000.],\
              [478.15, 2.511048614, 125000.],\
              [438.15, 2.064639971, 125000.],\
              [458.15, 2.064639971, 125000.],\
              [478.15, 2.064639971, 125000.],\
              [438.15, 1.618231329, 125000.],\
              [458.15, 1.618231329, 125000.],\
              [478.15, 1.618231329, 125000.]
]

#print(len(paraInList))
#print(paraInList)

# number of cases
NumOfCases = len(idxList)

# register each case name to a list of strings
Cases = []
for i in range(NumOfCases):
  s = "%03d"%idxList[i]
  Cases.append("C"+s)

# assertain each case's path
CasePath = []
for i in range(NumOfCases):
  path = CommonPath.joinpath(Cases[i]) 
  CasePath.append(path)
  #print(CasePath[i])

# loop over each case
for i in range(NumOfCases):
  VTMFileName = "case" + "%d"%idxList[i] + "_point.002000.vtm"

  VTMFilePath = CasePath[i].joinpath(Path(VTMFileName))

  # assertain each vtm file is alive
  #alive = VTMFilePath.exists()
  #print(alive)

  VTRFilePath = []

  # open and parsing the vtm files. Each case has one vtm file
  with open(VTMFilePath, "rb") as vtm:
    numOfBlock = 0

    while True:
      line = vtm.readline()
      if b"DataSet index" in line:
        numOfBlock += 1

        if idxList[i] < 10:
          VTRFilePath.append(line[32:48])
        elif 10<=idxList[i] and idxList[i]<100:
          VTRFilePath.append(line[32:49])
        elif idxList[i] >= 100:
          VTRFilePath.append(line[32:50])
        pass

      if not line:
        break
      pass  # over for 'while-loop'
    pass  # over, read vtm file

  # For certain case, loop all its vtr files, each of which relates to a block
  for j in range(numOfBlock):
    theVTRFile = CasePath[i].joinpath(VTRFilePath[j].decode("ASCII"))

    with open(theVTRFile, "rb") as vtr:
      line = vtr.readline() # version
      line = vtr.readline() # type and other
      line = vtr.readline() # grid type: RectilinearGrid
      line = vtr.readline() # Piece Extend

      # indexes range of i, j, k, from byte string to integer
      ista = int(line[17:22]); iend = int(line[22:27])
      jsta = int(line[27:32]); jend = int(line[32:37])
      ksta = int(line[37:42]); kend = int(line[42:47])

      line = vtr.readline()
      line = vtr.readline()
      line = vtr.readline()
      line = vtr.readline() # Cellvolume

      # names of each variable
      line = vtr.readline() # P
      varPBStr = line[38:39].decode("ASCII")

      line = vtr.readline() # U
      varUBStr = line[38:39].decode("ASCII")

      line = vtr.readline() # V
      varVBStr = line[38:39].decode("ASCII")

      line = vtr.readline() # W
      varWBStr = line[38:39].decode("ASCII")

      line = vtr.readline() # T
      varTBStr = line[38:39].decode("ASCII")

      line = vtr.readline() # /PointData
      line = vtr.readline() # /Coordinates

      line = vtr.readline()
      CoordX = line[38:45].decode("ASCII")

      line = vtr.readline()
      CoordY = line[38:45].decode("ASCII")

      line = vtr.readline()
      CoordZ = line[38:45].decode("ASCII")

      line = vtr.readline() # Coordinates
      line = vtr.readline() # Piece
      line = vtr.readline() # RectilinearGrid
      line = vtr.readline() # AppendedData ...

      FloatStartSymbol = vtr.read(1) # '_'

      # number of bytes for each float variables
      import numpy as np

      # ignore the Cellvolume
      numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
      vtr.seek(numOfBytes[0], 1)  # move forward with /numOfBytes/ bytes

      # field pressure
      numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
      numOfFloats = numOfBytes[0] // 8
      fieldP = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)

      # U field
      numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
      numOfFloats = numOfBytes[0] // 8
      fieldU = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)

      # V field
      numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
      numOfFloats = numOfBytes[0] // 8
      fieldV = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)

      # W field
      numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
      numOfFloats = numOfBytes[0] // 8
      fieldW = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)

      # T field
      numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
      numOfFloats = numOfBytes[0] // 8
      fieldT = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)

      # X coords
      numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
      numOfFloats = numOfBytes[0] // 8
      CoordsX = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)

      # Y coords
      numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
      numOfFloats = numOfBytes[0] // 8
      CoordsY = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)

      # Z coords
      numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
      numOfFloats = numOfBytes[0] // 8
      CoordsZ = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)

      #if i==0: print(CoordsZ)

