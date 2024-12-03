
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

  ista = line[17:22]; iend = line[22:27]
  jsta = line[27:32]; jend = line[32:37]
  ksta = line[37:42]; kend = line[42:47]

  #print(ista, len(ista), iend, len(iend))
  #print(jsta, len(jsta), jend, len(jend))
  #print(ksta, len(ksta), kend, len(kend))

# move all the numpy file into h5 data base



