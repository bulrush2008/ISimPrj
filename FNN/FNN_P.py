
"""
The FNN model is used to prediction Pressure field. And also as a base code
for Other field variables.

@author     @data       @aff        @version
Xia, S      24.12.11    Simpop.cn   v0.2
"""

# import libraries
import torch
import torch.nn as nn

import h5py
import pandas, numpy, random
import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

class FSimDataset(Dataset):
  def __init__(self, file:Path, caseList:list):
    """
    The data are remain store in h5 file, read when needed

    /caseList/: list of cases names, made of set either "test" or "train"
    """
    self.dataFile = h5py.File(file, 'r')
    self.caseList = caseList

  def __len__(self, idx):
    """
    return the input params and Pressure field
    """
    if idx >= self.num:
      raise IndexError

    data = []
    for blk in range(8):
      key = "Block-"+ "%02d"%blk + "-P"
      presFieldBlk = list(dataFile[caseList[i]][key][:])
      data += presFieldBlk

    return np.array(data)

    


