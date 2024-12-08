
'''
This module is for Pre, Store and Post submodules.
1, Pre  : read vtk file, transfer to .csv or numpy format
2, Store: Save the .csv file into h5 database.
3, Post : (1) read the data from database, with H5 format. and 
          (2) write the data a temporary vtk file, which would be displayed in GUI.

@author       @date         @aff          @version
Xia, S        2024.11.2     Simpop.cn     v0.1
'''

# read VTK, and transfer to numpy file

VTMFilePath = "D:\Development\FastSim\PSP\Results\Channel-Case1"
VTMFileName = "case1_point.002000.vtm"

dir = VTMFilePath+"\\"+VTMFileName

numOfBlocks = 0
fileLists = []

#with open(dir, 'r') as vtm:
with open(dir, 'rb') as vtm: # bytes string
  while True:
    line = vtm.readline()
    if line[13:18]==b"index":
      numOfBlocks += 1
      fileLists.append(line[32:48])
      #print(line[20:24], line[32:48])

    if not line:
      break

#print("All the blocks are ", numOfBlocks, "\n and they are ", fileLists)
#print(fileLists[0])

from pathlib import Path
curDir = Path.cwd()#; print(curDir, type(curDir))

VTRFilePath = curDir / "Results/Channel-Case1" / fileLists[5].decode('ascii')
print(VTRFilePath)

fileNameStr = str(VTRFilePath)
#print(fileNameStr)
#print(type(fileNameStr))
#fileNameStr = "D:\Development\FastSim\PSP\Results\LD1B\ld3d\\200\\1.vtr"

# check if the vtr file exists
import os
live = os.path.exists(fileNameStr)
#print(live, "Line 53")
#print(fileNameStr)

# The number of bytes offset is recorded at any time 
offset = 0

with open(fileNameStr, "rb") as vtr:
  line = vtr.readline(); offset += len(line)#; print(offset)
  line = vtr.readline(); offset += len(line)#; print(offset)
  line = vtr.readline(); offset += len(line)#; print(offset)
  line = vtr.readline(); offset += len(line)#; print(offset)

  # indexes of i, j, k
  istab = line[17:22]; iendb = line[22:27]
  jstab = line[27:32]; jendb = line[32:37]
  kstab = line[37:42]; kendb = line[42:47]

  # byte string change to int, no need to use char strings
  ista = int(istab); iend = int(iendb)
  jsta = int(jstab); jend = int(jendb)
  ksta = int(kstab); kend = int(kendb)

  #print(ista, iend)
  #print(jsta, jend)
  #print(ksta, kend)

  line = vtr.readline(); offset += len(line)
  line = vtr.readline(); offset += len(line)
  line = vtr.readline(); offset += len(line)

  # read the variable names
  # Cellvolume
  line = vtr.readline(); offset += len(line)
  var1 = line[38:48].decode("ASCII")
  #print(var1, type(var1))

  # P: pressure
  line = vtr.readline(); offset += len(line)
  var2 = line[38:39].decode("ASCII")
  #print(var2, type(var2))

  # U: 1st component of velocity
  line = vtr.readline(); offset += len(line)
  var3 = line[38:39].decode("ASCII")
  #print(var3, type(var3))

  # V: 2nd component of velocity
  line = vtr.readline(); offset += len(line)
  var4 = line[38:39].decode("ASCII")
  #print(var4, type(var4))

  # W: 3rd component of velocity
  line = vtr.readline(); offset += len(line)
  var5 = line[38:39].decode("ASCII")
  #print(var5, type(var5))

  # T: temperature
  line = vtr.readline(); offset += len(line)
  var6 = line[38:39].decode("ASCII")
  #print(var6, type(var6))

  line = vtr.readline(); offset += len(line)
  line = vtr.readline(); offset += len(line)

  # Xcenter
  line = vtr.readline(); offset += len(line)
  coordX = line[38:45].decode("ASCII")
  #print(coordX, type(coordX))

  # Ycenter
  line = vtr.readline(); offset += len(line)
  coordY = line[38:45].decode("ASCII")
  #print(coordY, type(coordY))

  # Zcenter
  line = vtr.readline(); offset += len(line)
  coordZ = line[38:45].decode("ASCII")
  #print(coordZ, type(coordZ))

  line = vtr.readline(); offset += len(line)
  line = vtr.readline(); offset += len(line)
  line = vtr.readline(); offset += len(line)
  line = vtr.readline(); offset += len(line)
  #print(line)
  #print("Now the offset is ", offset)

  #vtr.seek(offset+4)
  char = vtr.read(1).decode("ASCII")
  #print(char, "line 141")

  import numpy as np

  # first variable: Cellvolume
  dataNum = np.fromfile(vtr, dtype=np.int32, count=1)
  #print(dataNum)

  c = dataNum[0] // 8
  floats = np.fromfile(vtr, dtype=np.float64, count=c)

  #floats_csv = floats.reshape((27, 51, 12), order='F')
  #print(floats_csv[11, 33, 5])

  # P
  dataNum = np.fromfile(vtr, dtype=np.int32, count=1)
  #print(dataNum)

  c = dataNum[0] // 8
  floats = np.fromfile(vtr, dtype=np.float64, count=c)
  #print(floats[500:510])

  # U
  dataNum = np.fromfile(vtr, dtype=np.int32, count=1)
  #print(dataNum)

  c = dataNum[0] // 8
  floats = np.fromfile(vtr, dtype=np.float64, count=c)
  #print(floats[500:510])

  # V
  dataNum = np.fromfile(vtr, dtype=np.int32, count=1)
  #print(dataNum)

  c = dataNum[0] // 8
  floats = np.fromfile(vtr, dtype=np.float64, count=c)
  #print(floats[500:510])

  # W
  dataNum = np.fromfile(vtr, dtype=np.int32, count=1)
  #print(dataNum)

  c = dataNum[0] // 8
  floats = np.fromfile(vtr, dtype=np.float64, count=c)
  #print(floats[500:510])

  # T
  dataNum = np.fromfile(vtr, dtype=np.int32, count=1)
  #print(dataNum)

  c = dataNum[0] // 8
  floats = np.fromfile(vtr, dtype=np.float64, count=c)
  #print(floats[500:510])

  # coord x
  dataNum = np.fromfile(vtr, dtype=np.int32, count=1)
  #print(dataNum)

  c = dataNum[0] // 8
  floats = np.fromfile(vtr, dtype=np.float64, count=c)
  #print(floats[15])

  # coord y
  dataNum = np.fromfile(vtr, dtype=np.int32, count=1)
  #print(dataNum) # 51*8

  c = dataNum[0] // 8
  floats = np.fromfile(vtr, dtype=np.float64, count=c)
  #print(floats[15])

  # coord z
  dataNum = np.fromfile(vtr, dtype=np.int32, count=1)
  #print(dataNum) # 12*8

  c = dataNum[0] // 8
  floats = np.fromfile(vtr, dtype=np.float64, count=c)
  #print(floats[10])

  line = vtr.readline() # b'\n'
  #print(line)

  line = vtr.readline()
  #print(line)

  line = vtr.readline()
  print(line)


# transfer to numpy format

# move all the numpy file into h5 data base



