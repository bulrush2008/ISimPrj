
import h5py
from pathlib import Path

h5file = Path("../../FSCases/FSHDF/MatrixData.h5")
#alive = h5file.exists(); print(alive)

file = h5py.File(h5file, mode='r')

"""
# 列出文件中的所有对象
def print_attrs(name, obj):
  print(name)

file.visititems(print_attrs)
"""

# 查看输入参数
for group in file:
  print(file[group]["InParam"][:])

file.close()