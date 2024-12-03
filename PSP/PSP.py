
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

VTRFilePath = curDir / "Results/Channel-Case1" / fileLists[0].decode('ascii')
#print(VTRFilePath)
#print(type(VTRFilePath))

fileNameStr = str(VTRFilePath)
#print(fileNameStr)
#print(type(fileNameStr))

# check if the vtr file exists
#import os
#live = os.path.exists(fileNameStr)
#print(live)

with open(fileNameStr, "rb") as vtr:
  line = vtr.readline()
  line = vtr.readline()
  line = vtr.readline()
  line = vtr.readline()

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

  line = vtr.readline()
  line = vtr.readline()
  line = vtr.readline()

  # Cellvolume
  line = vtr.readline()
  var1Str = line[38:48].decode("ASCII")
  print(var1Str, type(var1Str))

  # P: pressure
  line = vtr.readline()
  var2Str = line[38:39].decode("ASCII")
  print(var2Str, type(var2Str))

  # U: 1st component of velocity
  line = vtr.readline()
  var3Str = line[38:39].decode("ASCII")
  print(var3Str, type(var3Str))

  # V: 2nd component of velocity
  line = vtr.readline()
  var4Str = line[38:39].decode("ASCII")
  print(var4Str, type(var4Str))

  # W: 3rd component of velocity
  line = vtr.readline()
  var5Str = line[38:39].decode("ASCII")
  print(var5Str, type(var5Str))

  # T: temperature
  line = vtr.readline()
  var6Str = line[38:39].decode("ASCII")
  print(var6Str, type(var6Str))

  # read the variable names

# move all the numpy file into h5 data base



