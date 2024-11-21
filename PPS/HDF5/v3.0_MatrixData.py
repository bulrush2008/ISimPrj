
'''
1, Create HDF5 file from known data
2, Read data from the HDF5 file

@author       @date         @aff          @version
Xia, Shuning  2024.11.20    Simpop.cn     v3.0
'''

import h5py
import numpy as np

# 假设有 5 个数据文件
data_files= ['Data01.dat', 'Data02.dat','Data03.dat','Data04.dat','Data05.dat']

#创建 HDF5 文件
with h5py.File("combined_data.h5", 'w') as hdf:
  # 遍历所有数据文件
  for idx, file in enumerate(data_files):
    # 假设文件中的数据是文本或 CSV 格式，读取成 numpy 数组
    data = np.loadtxt(file, delimiter=',') # 可以根据文件类型选择合适的读取方法

    # 将数据存储到 HDF5 文件中，每个数据文件一个数据集
    # 以 "dataset1", "dataset2", ... 作为数据集名称
    hdf.create_dataset(f'dataset{idx+1}', data=data)

print("HDF5 文件创建成功")

# 动态增量创建
newFile = "Data06.dat"

with h5py.File("combined_data.h5", 'a') as hdf:
  data = np.loadtxt(newFile, delimiter=',')

  hdf.create_dataset(f"{newFile}", data=data)

# 分层数据、异构数据

# 从 HDF5 文件读取数据

# 打开 HDF5 文件并读取数据
with h5py.File('combined_data.h5', 'r') as hdf:
    # 查看所有的数据集名称
    print("数据集列表:", list(hdf.keys()))
    
    # 假设读取第一个数据集
    dataset1 = hdf['dataset1'][:]
    print("dataset1 数据:\n", dataset1)

