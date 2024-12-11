
from pathlib import Path

import h5py

filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")

aLive = filePathH5.exists()
#print(aLive)

h5 = h5py.File(filePathH5, 'r')

print("len of h5 file: ", len(h5))

print("---- split ----")
print(h5.keys())

print("---- split ----")
"""
每个 case 有
  - 一个 input
  - 8 个 block
    - 5 个场变量：U/V/W/P/T
    - 3 个坐标数据
共有：(5+3)*8 + 1 = 65 个数据
"""
print("len of h5['C001']: ", len(h5["C001"]))

print("---- split ----")
print(h5["C001"].keys())