
from pathlib import Path

def WriteVTM(numOfBlocks:int, fileVTM:Path)->Path:
  vtm = open(fileVTM, 'wb')

  vtm.write(b'<?xml version="1.0"?>\n')
  vtm.write(b'<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">\n')
  vtm.write(b'  <vtkMultiBlockDataSet>\n')
  vtm.write(b'    <DataSet index="   1" file="channel/1.vtr"/>\n')
  vtm.write(b'    <DataSet index="   2" file="channel/2.vtr"/>\n')
  vtm.write(b'    <DataSet index="   3" file="channel/3.vtr"/>\n')
  vtm.write(b'    <DataSet index="   4" file="channel/4.vtr"/>\n')
  vtm.write(b'    <DataSet index="   5" file="channel/5.vtr"/>\n')
  vtm.write(b'    <DataSet index="   6" file="channel/6.vtr"/>\n')
  vtm.write(b'    <DataSet index="   7" file="channel/7.vtr"/>\n')
  vtm.write(b'    <DataSet index="   8" file="channel/8.vtr"/>\n')
  vtm.write(b'  </vtkMultiBlockDataSet>\n')
  vtm.write(b'</VTKFile>')

  vtm.close()

  return Path("channel")