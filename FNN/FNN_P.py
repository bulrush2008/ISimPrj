
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

    hdf = self.dataFile
    cid = self.caseList[idx]

    inp = hdf[cid]["InParam"][:]
    inp = torch.FloatTensor(inp)

    data = []
    for blk in range(8):
      key = "Block-"+ "%02d"%blk + "-P"

      presFieldBlk = list(hdf[cid][key][:])
      data += presFieldBlk
      pass

    return inp, torch.FloatTensor(data)
  
  def plotVTK(self, idx):
    if idx >= numOfCases:
      raise IndexError
    #TODO
    print("Todo: .plotVTK(...)")
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

#print(fsDataset_train.caseList)
#print(fsDataset_train.numCases)
#print(fsDataset_train.dataFile)

#p = fsDataset_train[1]

#print(type(p))
#print(len(p))
#print(p[100:109])

#fsDataset_train.plotVTK(1)

# Regression class: Core of FNN
class Regression(nn.Module):
  # 初始化 PyTorch 父类
  def __init__(self):
    super().__init__()

    # 初次设置 3 个隐藏层
    self.model = nn.Sequential(
      nn.Linear(3, 100),  # 3 inputs
      nn.LeakyReLU(0.02),
      nn.LayerNorm(100),

      nn.Linear(100,100),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(100),

      nn.Linear(100,125557), # output field, 8 block
      nn.Identity()
    )

    self.loss_function = nn.MSELoss()
    self.optimiser = torch.optim.SGD(self.parameters(),lr=0.01)

    self.counter = 0
    self.progress = []
    pass

  # 前向传播
  def forward(self, inputs):
    return self.model(inputs)

  # 训练
  def train(self, inputs, targets):
    outputs = self.forward(inputs)

    # 计算损失值
    loss = self.loss_function(outputs, targets)

    self.counter += 1
    if(self.counter%1 == 0):  # 对每个算例数据，记录损失值
      self.progress.append(loss.item())
      print(f"{self.counter} Cases Trained ...")
      pass

    # 梯度归零，反向传播，更新学习参数
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  # 打印损失函数
  def plot_progress(self):
    #TODO
    print("This obj.func() need implemented: plot_progress()")
    pass

  pass

R = Regression()

# train the model

epochs = 2

for i in range(epochs):
  print("Training Epoch", i+1, "of", epochs)

  ic = 0
  for bc, label in fsDataset_train:
    R.train(bc, label)

    ic += 1
    print(f"Case {ic} Trained")

  #for 
