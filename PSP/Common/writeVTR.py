
"""
write data into vtr files
"""

from pathlib import Path
import h5py
import numpy as np

def writeVTR(idxBlk:int, dirVTR:Path, dirHDF:Path, dataBounds:dict)->None:
  """
  Write a block's data to a VTR file, according to the input parameter 'idxBlk'.
  The data are from h5 file

  idxBlk: block index
  dirVTR: directory of the VTR file
  dirHDF: directory of source data with HDF5 format
  numCoordsEachBlk: number of coordinates of 3 axises for each vtr/block
  """

  # open the file of source data
  h5 = h5py.File(dirHDF, 'r')

  # One group saves data of one case 
  grpName = "FNN_Out"

  # first to read X/Y/Z, giving dims

  # X
  iSta = dataBounds["X"][0]
  iEnd = dataBounds["X"][1]
  lenX = iEnd - iSta

  # Y
  iSta = dataBounds["Y"][0]
  iEnd = dataBounds["Y"][1]
  lenY = iEnd - iSta

  # Z

  iSta = dataBounds["Z"][0]
  iEnd = dataBounds["Z"][1]
  lenZ = iEnd - iSta

  # define the dims: start and final index
  iSta = 1; iEnd = lenX
  jSta = 1; jEnd = lenY
  kSta = 1; kEnd = lenZ

  extent  = str("%5d"%iSta)  \
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
  bStrs = bStr1 + bStr2 + bStr3
  vtr.write(bStrs)

  bStr1 = b'  <Piece Extent="'
  bStr2 = extentByte
  bStr3 = b'">\n'
  bStrs = bStr1 + bStr2 + bStr3
  vtr.write(bStrs)

  vtr.write(b'    <CellData>\n')
  vtr.write(b'    </CellData>\n')
  vtr.write(b'    <PointData>\n')

  # write P/U/V/W/T field and X/Y/Z

#   # header - for P
#   OffsetP = 0 # the first variable, no offsets
#   bStr1 = b'      <DataArray type="Float64" Name="P" NumberOfComponents="1" format="appended" offset="'
#   bStr2 = str("%10d"%OffsetP).encode("utf-8")
#   bStr3 = b'"/>\n'
#   bStrs = bStr1 + bStr2 + bStr3
#   #print(bStrs)
#   vtr.write(bStrs)

  # header - for U
  bStr1 = b'      <DataArray type="Float64" Name="U" NumberOfComponents="1" format="appended" offset="'

  # OffsetU = OffsetP + lenX*lenY*lenZ * 8 + 4  # P before
  OffsetU = 0
  bStr2 = str("%10d"%OffsetU).encode("utf-8")

  bStr3 = b'"/>\n'
  bStrs = bStr1 + bStr2 + bStr3
  vtr.write(bStrs)

  # header - for V
  bStr1 = b'      <DataArray type="Float64" Name="V" NumberOfComponents="1" format="appended" offset="'

  # OffsetV = OffsetU * 2 # P/U before
  OffsetV = OffsetU + lenX*lenY*lenZ*8 + 4
  bStr2 = str("%10d"%OffsetV).encode("utf-8")
  bStr3 = b'"/>\n'
  bStrs = bStr1 + bStr2 + bStr3
  vtr.write(bStrs)

  # header - for W
  bStr1 = b'      <DataArray type="Float64" Name="W" NumberOfComponents="1" format="appended" offset="'
  # OffsetW = OffsetU * 3 # P/U/V before
  OffsetW = OffsetV + lenX*lenY*lenZ*8 + 4
  bStr2 = str("%10d"%OffsetW).encode("utf-8")
  bStr3 = b'"/>\n'
  bStrs = bStr1 + bStr2 + bStr3
  vtr.write(bStrs)

  # # header - for T
  # bStr1 = b'      <DataArray type="Float64" Name="T" NumberOfComponents="1" format="appended" offset="'
  # OffsetT = OffsetU * 4 # P/U/V/W before
  # bStr2 = str("%10d"%OffsetT).encode("utf-8")	
  # bStr3 = b'"/>\n'
  # bStrs = bStr1 + bStr2 + bStr3
  # vtr.write(bStrs)

  vtr.write(b'    </PointData>\n')

  # headers for coordinates,  they are necessary!
  vtr.write(b'    <Coordinates>\n')

  # header - X
  bStr1 = b'      <DataArray type="Float64" Name="Xcenter" NumberOfComponents="1" format="appended" offset="'

  # OffsetX = OffsetU * 5 # P/U/V/W/T before
  OffsetX = OffsetW + lenX*lenY*lenZ*8 + 4
  bStr2 = str("%10d"%OffsetX).encode("utf-8")
  bStr3 = b'"/>\n'
  bStrs = bStr1 + bStr2 + bStr3
  vtr.write(bStrs)

  # header - Y
  bStr1 = b'      <DataArray type="Float64" Name="Ycenter" NumberOfComponents="1" format="appended" offset="'
  OffsetY = OffsetX + lenX*8 + 4 # P/U/V/W/T/X before
  bStr2 = str("%10d"%OffsetY).encode("utf-8")
  bStr3 = b'"/>\n'
  bStrs = bStr1 + bStr2 + bStr3
  vtr.write(bStrs)

  # header - Z
  bStr1 = b'      <DataArray type="Float64" Name="Zcenter" NumberOfComponents="1" format="appended" offset="'
  OffsetZ = OffsetY + lenY*8 + 4 # P/U/V/W/T/X/Y before
  bStr2 = str("%10d"%OffsetZ).encode("utf-8")
  bStr3 = b'"/>\n'
  bStrs = bStr1 + bStr2 + bStr3
  vtr.write(bStrs)

  vtr.write(b'    </Coordinates>\n')
  vtr.write(b'  </Piece>\n')
  vtr.write(b'  </RectilinearGrid>\n')
  vtr.write(b'  <AppendedData encoding="raw">\n')

  #----------------------------------------------------------------------------
  # write data

  # add '_' denoting the starting floats data
  vtr.write(b'_')

  iSta = dataBounds["Var"][0]
  iEnd = dataBounds["Var"][1]

#   # write pressure field "P"
#   datasetName = "P"

#   fieldP = h5[grpName][datasetName][iSta:iEnd]
#   numOfBytes = np.int32((len(fieldP)*8))

#   # write the byte offsets and float loop data
#   numOfBytes.tofile(vtr)
#   fieldP.tofile(vtr)

  # write U field "U"
  datasetName = "U"

  fieldU = h5[grpName][datasetName][iSta:iEnd]
  numOfBytes = np.int32(len(fieldU)*8)

  numOfBytes.tofile(vtr)
  fieldU.tofile(vtr)

  # write pressure field "V"
  datasetName = "V"

  fieldV = h5[grpName][datasetName][iSta:iEnd]
  numOfBytes = np.int32((len(fieldV)*8))

  # write the byte offsets and float loop data
  numOfBytes.tofile(vtr)
  fieldV.tofile(vtr)

  # write pressure field "W"
  datasetName = "W"

  fieldW = h5[grpName][datasetName][iSta:iEnd]
  numOfBytes = np.int32((len(fieldW)*8))

  # write the byte offsets and float loop data
  numOfBytes.tofile(vtr)
  fieldW.tofile(vtr)

#   # write pressure field "T"
#   datasetName = "T"

#   fieldT = h5[grpName][datasetName][iSta:iEnd]
#   numOfBytes = np.int32((len(fieldT)*8))

#   # write the byte offsets and float loop data
#   numOfBytes.tofile(vtr)
#   fieldT.tofile(vtr)

  # write coords X
  datasetName = "Coords-X"

  iSta = dataBounds["X"][0]
  iEnd = dataBounds["X"][1]
  coordsX = h5[grpName][datasetName][iSta:iEnd]
  numOfBytes = np.int32(len(coordsX)*8)

  numOfBytes.tofile(vtr)
  coordsX.tofile(vtr)

  # write coords Y
  datasetName = "Coords-Y"

  iSta = dataBounds["Y"][0]
  iEnd = dataBounds["Y"][1]
  coordsY = h5[grpName][datasetName][iSta:iEnd]
  numOfBytes = np.int32(len(coordsY)*8)

  numOfBytes.tofile(vtr)
  coordsY.tofile(vtr)

  # write coords Z
  datasetName = "Coords-Z"

  iSta = dataBounds["Z"][0]
  iEnd = dataBounds["Z"][1]
  coordsZ = h5[grpName][datasetName][iSta:iEnd]
  numOfBytes = np.int32(len(coordsZ)*8)

  numOfBytes.tofile(vtr)
  coordsZ.tofile(vtr)

  vtr.write(b'\n')
  vtr.write(b'  </AppendedData>\n')
  vtr.write(b'</VTKFile>')
  pass