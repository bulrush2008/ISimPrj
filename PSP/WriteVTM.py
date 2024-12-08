
from pathlib import Path

def WriteVTM(numOfBlocks:int, fileVTM:Path)->Path:
  vtm = open(fileVTM, 'wb')
  #vtm.writelines(b"<?xml version=\"1.0\"?>")

  vtm.close()
  pass