
from pathlib import Path

def WriteVTM(numOfBlocks:int, fileVTM:Path)->Path:
  vtm = open(fileVTM, 'wb')
  vtm.write(b'<?xml version="1.0"?>\n')
  vtm.write(b'<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">\n')
  vtm.write(b'  <vtkMultiBlockDataSet>\n')
  vtm.write(b'    <DataSet index="   1" file="channel/1.vtr"/>\n')
  vtm.write(b'  </vtkMultiBlockDataSet>\n')
  vtm.write(b'</VTKFile>')

  vtm.close()

  return Path("channel")