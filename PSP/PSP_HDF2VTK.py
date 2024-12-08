
"""
Get data from .h5 file, write them to vtm and vtr files, in order to be
  displayed by Paraview

@author       @date         @aff          @version
Xia, S        2024.11.8     Simpop.cn     v1.0
"""

from pathlib import Path

from WriteVTM import WriteVTM

numOfBlocks = 8
dirVTM = Path("./Disp")
if not dirVTM.exists(): dirVTM.mkdir()
#if dirVTM.exists(): dirVTM.rmdir() # 删除一个非空目录

fileVTM = dirVTM.joinpath("t01.vtm")

# todo
dirVTR = WriteVTM(numOfBlocks, fileVTM)

dirVTR = dirVTM.joinpath("blocks/")
print(dirVTR)