
import h5py
import numpy as np

# 假设有 5 个数据文件
data_files= ['Data01.dat', 'Data02.dat','Data03.dat','Data04.dat','Data05.dat']

#创建 HDF5 文件
with h5py.File("combined_data.h5", 'w') as hdf:
  #print("debug, line 10")
  # 遍历所有文件
  for idx, file in enumerate(data_files):
    #print(f"idx={idx}, file={file}")
    data = np.loadtxt(file, delimiter=',')

    hdf.create_dataset(f'dataset{idx+1}', data=data)

print("HDF5 文件创建成功")



# TODO: 动态增量创建