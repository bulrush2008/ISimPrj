import h5py
import torch
import numpy as np
from pathlib import Path

def write2HDF(pred:torch.Tensor, eval_dir_path:Path, var_name:list[str], coords:dict=None):
#-----------------------------------------------------------------------------
    """
    - 将预测数据，写入 HDF 数据库
    - 如有必要，会写入坐标
    """
    # h5 文件已经在外部打开，这里只需要创建一个组，用来管理模型的预测数据即可
    grp_name = "FNN_Out" # a case data is a group

    # Create parent directories if they don't exist
    eval_dir_path.parent.mkdir(parents=True, exist_ok=True)

    # 以附加的方式打开
    h5 = h5py.File(eval_dir_path, 'a')

    # 不知道某变量流场是否已经写入，需要检测、区分
    if grp_name in h5:
        grp = h5[grp_name] # if the group existed
    else:
        grp = h5.create_group(grp_name)  # if not, created it
        pass

    # the predicted data should be detached and converted to numpy format
    pred = pred.detach().cpu().numpy() # [NUM_POINTS, NUM_VARS]

    # write data into h5 database directly
    for i,var in enumerate(var_name):
        ds_name = f"{var}"
        data = pred[:,i].flatten()
        grp.create_dataset(ds_name, data=data, dtype=np.float64)

    # write coordinates it necessary
    if coords is not None:
        ds_name = "Coords-X"
        grp.create_dataset(ds_name, data=coords["x"], dtype=np.float64)

        ds_name = "Coords-Y"
        grp.create_dataset(ds_name, data=coords["y"], dtype=np.float64)

        ds_name = "Coords-Z"
        grp.create_dataset(ds_name, data=coords["z"], dtype=np.float64)
        pass

    h5.close()
    pass
