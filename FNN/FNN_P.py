
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
    - The data are remain store in h5 file, read when needed

    inputs:
    /caseList/: list of cases names, made of set either "test" or "train"
    """
    self.dataFile = h5py.File(file, 'r')
    self.caseList = caseList
    self.numCases = len(caseList)

  def __len__(self):
    return self.numCases

  def __getitem__(self, idx):
    """
    return the input params and Pressure field
    """
    if idx >= numOfCases:
      raise IndexError

    data = []
    for blk in range(8):
      key = "Block-"+ "%02d"%blk + "-P"
      presFieldBlk = list(dataFile[caseList[i]][key][:])
      data += presFieldBlk
      pass

    return np.array(data)
  
  def plotVTK(self, index):
    if idx >= numOfCases:
      raise IndexError
    #TODO
    pass

# split the data, 27 = 22 + 5

numOfCases = 27
ratioTest = 0.2

sizeOfTestSet = np.int64(numOfCases * ratioTest)

np.random.seed(42)
permut = np.random.permutation(numOfCases)

from idxList import idxList

listTestCase = []
for i in permut[:sizeOfTestSet]:
  theCase = "C" + "%03d"%idxList[i]
  listTestCase.append(theCase)

#print(listTestCase, len(listTestCase))

listTrainCase = []
for i in permut[sizeOfTestSet:]:
  theCase = "C" + "%03d"%idxList[i]
  listTrainCase.append(theCase)

#print(listTrainCase)
#print(len(listTrainCase))
#print(type(listTrainCase))

from pathlib import Path
import h5py

filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")

#aLive = filePathH5.exists()
#print(aLive)

fsDataset_train = FSimDataset(filePathH5, listTrainCase)




