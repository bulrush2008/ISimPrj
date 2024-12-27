
"""
Get data from .h5 file, write them to vtm and vtr files, in order to be
  displayed by Paraview

@author       @date         @aff          @version
Xia, S        2024.11.8     Simpop.cn     v1.0
"""

from pathlib import Path

from writeVTM import writeVTM
from writeVTR import writeVTR

numOfBlocks = 8
dirVTM = Path("./FNN_Case_Test")
if not dirVTM.exists(): dirVTM.mkdir()
#if dirVTM.exists(): dirVTM.rmdir() # 删除一个空目录

fileVTM = dirVTM.joinpath("t01.vtm")

# write the vtm file and return the path of vtr files
dirVTR = writeVTM(numOfBlocks, fileVTM)
#print(dirVTR)

dirVTR = dirVTM.joinpath(dirVTR)
print(dirVTR)

if not dirVTR.exists(): dirVTR.mkdir(parents=True)

dirHDF = Path("../FNN").joinpath("fnn.h5")
alive = dirHDF.exists()
print("HDF File exists? ", alive)

for idx in range(numOfBlocks):
  writeVTR(idx, dirVTR, dirHDF)
