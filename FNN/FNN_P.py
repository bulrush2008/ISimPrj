
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
  def __init__(self, file:Path, idxList:np.ndarray):
    self.fileObj = h5py.File(file, 'r')
    self.dataset = fileObj["C001"]  # the first data