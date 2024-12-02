
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

# 从 HDF5 文件读取数据



# TODO: 动态增量创建