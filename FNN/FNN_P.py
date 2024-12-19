
"""
The FNN model is used to prediction Pressure field. And also as a base code
for Other field variables.

@author     @data       @aff        @version
Xia, S      24.12.19    Simpop.cn   v2.x
"""

# import libraries
import torch
from torch.utils.data import Dataset

import h5py
import pandas, numpy, random
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np

from Common.idxList import idxList
from Common.idxList import numOfAllCases

from Common.Regression import Regression

class FSimDataset(Dataset):
  def __init__(self, file:Path, caseList:list):
    """
    - The data remain stored in h5 file, read when needed

    - inputs:
    /caseList/: list of cases names, made of set either "test" or "train"

    - member data: self.x
      - dataFile: HDF5 file
      - caseList: dataList, a char string list in hdf5 file
      - numCases: number of cases input
    """
    self.dataFile = h5py.File(file, 'r')
    self.caseList = caseList
    self.numCases = len(caseList)

  def __len__(self):
    return self.numCases

  def __getitem__(self, idx):
    """
    return the input params and field
    """
    if idx >= numOfCases:
      raise IndexError

    hdf = self.dataFile
    cid = self.caseList[idx]

    inp = hdf[cid]["InParam"][:]
    inp = torch.FloatTensor(inp)

    data = []
    coords = {}

    coords["x"] = [[]]
    coords["y"] = [[]]
    coords["z"] = [[]]

    for blk in range(8):
      key = "Block-"+ "%02d"%blk + "-P"

      presFieldBlk = list(hdf[cid][key][:])
      data += presFieldBlk

      # coordx
      key = "Block-"+ "%02d"%blk + "-X"
      crd = list(hdf[cid][key][:])
      coords["x"].append(crd)

      # coordy
      key = "Block-"+ "%02d"%blk + "-Y"
      crd = list(hdf[cid][key][:])
      coords["y"].append(crd)

      # coordz
      key = "Block-"+ "%02d"%blk + "-Z"
      crd = list(hdf[cid][key][:])
      coords["z"].append(crd)
      pass

    del coords["x"][0]
    del coords["y"][0]
    del coords["z"][0]

    return inp, torch.FloatTensor(data), coords
  
  def plotVTK(self, idx):
    pass

# split the data, 49 = 40 + 9
numOfCases = numOfAllCases
ratioTest = 0.2

sizeOfTestSet = np.int64(numOfCases * ratioTest)

# 42 是随机种子，其它整数也可以
np.random.seed(42)
permut = np.random.permutation(numOfCases)

listTestCase = []
for i in permut[:sizeOfTestSet]:
  theCase = "C" + "%03d"%idxList[i]
  listTestCase.append(theCase)

listTrainCase = []
for i in permut[sizeOfTestSet:]:
  theCase = "C" + "%03d"%idxList[i]
  listTrainCase.append(theCase)

filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")

#aLive = filePathH5.exists()
#print(aLive)

fsDataset_train = FSimDataset(filePathH5, listTrainCase)


# 生成一个回归模型对象
R = Regression()

# train the model
epochs = 1

for i in range(epochs):
  print("Training Epoch", i+1, "of", epochs)

  for bc, label, _ in fsDataset_train:
    #print(bc)
    R.train(bc, label)
    pass
  pass

# 绘制损失函数历史
DirPNG = Path("./Pics")
R.saveLossHistory2PNG(DirPNG)

# 预测

fsDataset_test = FSimDataset(filePathH5, listTestCase)
#print(listTestCase)

#len_test = fsDataset_test.numCases
# for C034
inp, pField, coords = fsDataset_test[0]
#print(fsDataset_test.caseList)
#print(type(inp), type(pField))

#outPresTorch = R.forward(inp)

#outPres = outPresTorch.detach().numpy()

#print(outPres[100])
#print(type(outPres))

R.write2HDF(inp, Path("./fnn.h5"),coords=coords)

