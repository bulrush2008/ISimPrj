
from idxList import idxList

def ReadVTM(VTMFilePath, i):
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
  return numOfBlock, VTRFilePath