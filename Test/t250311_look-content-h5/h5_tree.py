
import h5py
import os
from pathlib import  Path

def print_hdf5_tree(file, indent=0, last=True, header=''):
    """递归打印HDF5文件结构"""
    # 设置树形连接符号
    branch = '└─ ' if last else '├─ '
    connector = '    ' if last else '│   '

    # 打印当前节点信息
    if isinstance(file, h5py.File):
        print(header + branch + os.path.basename(file.filename))
    elif isinstance(file, h5py.Group):
        print(header + branch + f"(G) {file.name.split('/')[-1]} [{len(file.attrs)} attrs]")
    elif isinstance(file, h5py.Dataset):
        dtype = file.dtype
        shape = file.shape
        print(header + branch + f"(D) {file.name.split('/')[-1]} {shape} ({dtype})")

    # 递归处理子节点
    if isinstance(file, (h5py.File, h5py.Group)):
        items = list(file.items())
        for i, (name, obj) in enumerate(items):
            is_last = i == len(items)-1
            print_hdf5_tree(obj, indent+1, is_last, header + connector)

# 使用示例
with h5py.File(Path('./temp.h5'), 'r') as f:
    print_hdf5_tree(f)