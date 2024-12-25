
from Common.idxList import idxList

from pathlib import Path

def readVTM(VTMFilePath:Path, idxi:int)->(int, list):
  VTRFilePath = []

  # open and parsing the vtm files. Each case has one vtm file
  with open(VTMFilePath, "rb") as vtm:
    numOfBlock = 0

    while True:
      line = vtm.readline()
      if b"DataSet index" in line:
        numOfBlock += 1

        if idxi < 10:
          VTRFilePath.append(line[32:48])
        elif 10<=idxi and idxi<100:
          VTRFilePath.append(line[32:49])
        elif idxi >= 100:
          VTRFilePath.append(line[32:50])
        pass

      if not line:
        break
      pass  # over for 'while-loop'
    pass  # over, read vtm file
  return numOfBlock, VTRFilePath