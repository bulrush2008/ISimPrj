
import numpy as np
from pathlib import Path

def readVTR(theVTRFile:Path)->tuple:
#------------------------------------------------------------------------------
  """
  read each VTR file, get the field data and coords, and then return
  - theVTRFile: the VTR file's path
  - return: a tuple including each field and coords, each is of type np.ndarray
    (fieldP, fieldU, fieldV, fieldW, fieldT, coordsX, coordsY, coordsZ)
  """

  #----------------------------------------------------------------------------
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

    # float data started with an underline "_"
    underline = vtr.read(1) 

    # ignore the Cellvolume
    numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
    vtr.seek(numOfBytes[0], 1)  # move forward with /numOfBytes/ bytes

    # P field
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
    coordsX = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)

    # Y coords
    numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
    numOfFloats = numOfBytes[0] // 8
    coordsY = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)

    # Z coords
    numOfBytes = np.fromfile(vtr, dtype=np.int32, count=1)  # byte offsets
    numOfFloats = numOfBytes[0] // 8
    coordsZ = np.fromfile(vtr, dtype=np.float64, count=numOfFloats)
    pass

  return fieldP, fieldU, fieldV, fieldW, fieldT, coordsX, coordsY, coordsZ