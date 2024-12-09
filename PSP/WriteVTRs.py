
"""
write data into vtr files
"""

from pathlib import Path
import h5py
import numpy as np

def WriteVTRs(idxBlk:int, dirVTR:Path, dirHDF:Path)->None:
  """
  Write a block's data to a VTR file, according to the input parameter 'idxBlk'.
  The data are from h5 file
  """
  h5 = h5py.File(dirHDF, 'r')

  # take "C001" as an example. Other case can be selected too. 
  grpName = "C001"

  # first to read X/Y/Z, giving dims

  # X
  datasetName = "Block-" + "%02d"%idxBlk + "-X"
  coordsX = h5[grpName][datasetName][:]

  lenX = len(coordsX)#; print("lenX = ", lenX)

  # Y
  datasetName = "Block-" + "%02d"%idxBlk + "-Y"
  coordsY = h5[grpName][datasetName][:]

  lenY = len(coordsY)#; print("lenY = ", lenY)

  # Z
  datasetName = "Block-" + "%02d"%idxBlk + "-Z"
  coordsZ = h5[grpName][datasetName][:]

  lenZ = len(coordsZ)#; print("lenZ = ", lenZ)

  # define the dims: start and final index
  iSta = 1; iEnd = lenX#; print(iSta, iEnd)
  jSta = 1; jEnd = lenY#; print(jSta, jEnd)
  kSta = 1; kEnd = lenZ#; print(kSta, kEnd)

  extent = str("%5d"%iSta)  \
         + str("%5d"%iEnd)  \
         + str("%5d"%jSta)  \
         + str("%5d"%jEnd)  \
         + str("%5d"%kSta)  \
         + str("%5d"%kEnd)

  #print(extent, type(extent))

  extentByte = extent.encode("utf8")
  #print(extentByte)

  # the block's name
  fileNameVTR = "%d"%(idxBlk+1) + ".vtr"
  filePathVTR = dirVTR.joinpath(fileNameVTR)

  vtr = open(filePathVTR, 'wb')

  # write header lines
  vtr.write(b'<?xml version="1.0"?>\n')
  vtr.write(b'<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n')

  bStr1 = b'  <RectilinearGrid WholeExtent="'
  bStr2 = extentByte
  bStr3 = b'">\n'
  vtr.write(bStr1 + bStr2 + bStr3)

  bStr1 = b'  <Piece Extent="'
  bStr2 = extentByte
  bStr3 = b'">\n'
  vtr.write(bStr1 + bStr2 + bStr3)

  vtr.write(b'    <CellData>\n')
  vtr.write(b'    </CellData>\n')
  vtr.write(b'    <PointData>\n')

  # write P/U/V/W/T field and X/Y/Z
  vtr.write(b'      <DataArray type="Float64" Name="P" NumberOfComponents="1" format="appended" offset="     0"/>\n')
  vtr.write(b'      <DataArray type="Float64" Name="U" NumberOfComponents="1" format="appended" offset="116692"/>\n')
  vtr.write(b'      <DataArray type="Float64" Name="V" NumberOfComponents="1" format="appended" offset="233384"/>\n')
  vtr.write(b'      <DataArray type="Float64" Name="W" NumberOfComponents="1" format="appended" offset="350076"/>\n')
  vtr.write(b'      <DataArray type="Float64" Name="T" NumberOfComponents="1" format="appended" offset="466768"/>\n')
  vtr.write(b'    </PointData>\n')

  # coordinates are necessary
  vtr.write(b'    <Coordinates>\n')
  vtr.write(b'      <DataArray type="Float64" Name="Xcenter" NumberOfComponents="1" format="appended" offset=" 583460"/>\n')
  vtr.write(b'      <DataArray type="Float64" Name="Ycenter" NumberOfComponents="1" format="appended" offset=" 583672"/>\n')
  vtr.write(b'      <DataArray type="Float64" Name="Zcenter" NumberOfComponents="1" format="appended" offset=" 584084"/>\n')
  vtr.write(b'    </Coordinates>\n')
  vtr.write(b'  </Piece>\n')
  vtr.write(b'  </RectilinearGrid>\n')
  vtr.write(b'  <AppendedData encoding="raw">\n')

  # add '_' denoting the starting floats data
  vtr.write(b'_')

  # write pressure field "P"
  datasetName = "Block-" + "%02d"%idxBlk + "-P"

  fieldP = h5[grpName][datasetName][:]  # "[:]" is necessary
  numOfBytes = np.int32((len(fieldP)*8))

  # write the byte offsets and float loop data
  numOfBytes.tofile(vtr)
  fieldP.tofile(vtr)

  # write U field "U"
  datasetName = "Block-" + "%02d"%idxBlk + "-U"

  fieldU = h5[grpName][datasetName][:]
  numOfBytes = np.int32(len(fieldU)*8)

  numOfBytes.tofile(vtr)
  fieldU.tofile(vtr)

  # write pressure field "V"
  datasetName = "Block-" + "%02d"%idxBlk + "-V"

  fieldV = h5[grpName][datasetName][:]  # "[:]" is necessary
  numOfBytes = np.int32((len(fieldV)*8))

  # write the byte offsets and float loop data
  numOfBytes.tofile(vtr)
  fieldV.tofile(vtr)

  # write pressure field "W"
  datasetName = "Block-" + "%02d"%idxBlk + "-W"

  fieldW = h5[grpName][datasetName][:]  # "[:]" is necessary
  numOfBytes = np.int32((len(fieldW)*8))

  # write the byte offsets and float loop data
  numOfBytes.tofile(vtr)
  fieldW.tofile(vtr)

  # write pressure field "T"
  datasetName = "Block-" + "%02d"%idxBlk + "-T"

  fieldT = h5[grpName][datasetName][:]  # "[:]" is necessary
  numOfBytes = np.int32((len(fieldT)*8))

  # write the byte offsets and float loop data
  numOfBytes.tofile(vtr)
  fieldT.tofile(vtr)

  # write coords X
  datasetName = "Block-" + "%02d"%idxBlk + "-X"

  coordsX = h5[grpName][datasetName][:]
  numOfBytes = np.int32(len(coordsX)*8)

  numOfBytes.tofile(vtr)
  coordsX.tofile(vtr)

  # write coords Y
  datasetName = "Block-" + "%02d"%idxBlk + "-Y"

  coordsY = h5[grpName][datasetName][:]
  numOfBytes = np.int32(len(coordsY)*8)

  numOfBytes.tofile(vtr)
  coordsY.tofile(vtr)

  # write coords Z
  datasetName = "Block-" + "%02d"%idxBlk + "-Z"

  coordsZ = h5[grpName][datasetName][:]
  numOfBytes = np.int32(len(coordsZ)*8)

  numOfBytes.tofile(vtr)
  coordsZ.tofile(vtr)

  vtr.write(b'\n')
  vtr.write(b'  </AppendedData>\n')
  vtr.write(b'</VTKFile>')
  pass