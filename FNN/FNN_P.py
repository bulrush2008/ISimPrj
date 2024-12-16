
"""
The FNN model is used to prediction Pressure field. And also as a base code
for Other field variables.

@author     @data       @aff        @version
Xia, S      24.12.12    Simpop.cn   v1.0
"""

# import libraries
import torch
import torch.nn as nn

import h5py
import pandas, numpy, random
import matplotlib.pyplot as plt

import math

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

listTrainCase = []
for i in permut[sizeOfTestSet:]:
  theCase = "C" + "%03d"%idxList[i]
  listTrainCase.append(theCase)

from pathlib import Path
import h5py

filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")

#aLive = filePathH5.exists()
#print(aLive)

fsDataset_train = FSimDataset(filePathH5, listTrainCase)

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

      nn.Linear(100,50),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(50),

      nn.Linear(50,25),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(25),

      nn.Linear(25,10),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(10),

      nn.Linear(10,25),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(25),

      nn.Linear(25,50),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(50),

      nn.Linear(50,100),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(100),

      nn.Linear(100,125557), # output field, 8 block
      nn.Identity()
    )

    # 初始化权重，目前使用 He Kaiming 方法
    self._initialize_weights()

    # 回归问题，需要使用 MSE
    self.loss_function = nn.MSELoss()
    self.optimiser = torch.optim.SGD(self.parameters(),lr=0.0001)

    # counter 用来记录训练的次数
    self.counter = 0
    self.progress = []  # 存储每一步的损失值
    pass

  def _initialize_weights(self):
    """
    内部函数，在模型初始化时调用一次，用于配置初始化参数
    """
    for m in self.model:
      if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  # 前向传播
  def forward(self, inputs):
    return self.model(inputs)

  # 训练
  def train(self, inputs, targets):
    outputs = self.forward(inputs)

    # 计算损失值
    loss = self.loss_function(outputs, targets)

    # each train step, the loss must be added
    self.progress.append(loss.item())

    self.counter += 1

    if(self.counter%10 == 0):  # 对每个算例数据，记录损失值
      print(f"{self.counter} Cases Trained ...")
      pass

    # 梯度归零，反向传播，更新学习参数
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  def write2HDF(self, inp:torch.FloatTensor, dirFileHDF:Path, coords:list=None):
    h5 = h5py.File(dirFileHDF, 'w')

    grpName = "FNN_Out" # 相当于原“每个 Case”
    grp = h5.create_group(grpName)

    # 将预测数据，转化为 numpy 矩阵数据
    output = self.forward(inp).detach().numpy()

    ptsB1 = [2,27,2,52,2,12]
    numbPtsB1 = (ptsB1[1]-ptsB1[0]+1) * (ptsB1[3]-ptsB1[2]+1) * (ptsB1[5]-ptsB1[4]+1)

    ptsB2 = [2,27,2,52,2,13]
    numbPtsB2 = (ptsB2[1]-ptsB2[0]+1) * (ptsB2[3]-ptsB2[2]+1) * (ptsB2[5]-ptsB2[4]+1)

    ptsB3 = [2,27,2,53,2,12]
    numbPtsB3 = (ptsB3[1]-ptsB3[0]+1) * (ptsB3[3]-ptsB3[2]+1) * (ptsB3[5]-ptsB3[4]+1)

    ptsB4 = [2,27,2,53,2,13]
    numbPtsB4 = (ptsB4[1]-ptsB4[0]+1) * (ptsB4[3]-ptsB4[2]+1) * (ptsB4[5]-ptsB4[4]+1)

    ptsB5 = [2,28,2,52,2,12]
    numbPtsB5 = (ptsB5[1]-ptsB5[0]+1) * (ptsB5[3]-ptsB5[2]+1) * (ptsB5[5]-ptsB5[4]+1)

    ptsB6 = [2,28,2,52,2,13]
    numbPtsB6 = (ptsB6[1]-ptsB6[0]+1) * (ptsB6[3]-ptsB6[2]+1) * (ptsB6[5]-ptsB6[4]+1)

    ptsB7 = [2,28,2,53,2,12]
    numbPtsB7 = (ptsB7[1]-ptsB7[0]+1) * (ptsB7[3]-ptsB7[2]+1) * (ptsB7[5]-ptsB7[4]+1)

    ptsB8 = [2,28,2,53,2,13]
    numbPtsB8 = (ptsB8[1]-ptsB8[0]+1) * (ptsB8[3]-ptsB8[2]+1) * (ptsB8[5]-ptsB8[4]+1)

    numbAll = numbPtsB1 + numbPtsB2 + numbPtsB3 + numbPtsB4 \
            + numbPtsB5 + numbPtsB6 + numbPtsB7 + numbPtsB8
    #print(numbAll)

    # for block 1
    idxB = 0
    dsName = "Block-" + "%02d"%idxB + "-P"

    ista = 0; iend = numbPtsB1
    grp.create_dataset(dsName, data=output[ista:iend], dtype=np.float64)

    # 写入坐标
    if coords is not None:
      dsName = "Block-" + "%02d"%idxB + "-X"
      grp.create_dataset(dsName, data=coords["x"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Y"
      grp.create_dataset(dsName, data=coords["y"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Z"
      grp.create_dataset(dsName, data=coords["z"][idxB], dtype=np.float64)
      pass

    # for block 2
    idxB = 1
    dsName = "Block-" + "%02d"%idxB + "-P"

    ista = numbPtsB1; iend = numbPtsB1 + numbPtsB2
    grp.create_dataset(dsName, data=output[ista:iend], dtype=np.float64)

    # 写入坐标
    if coords is not None:
      dsName = "Block-" + "%02d"%idxB + "-X"
      grp.create_dataset(dsName, data=coords["x"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Y"
      grp.create_dataset(dsName, data=coords["y"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Z"
      grp.create_dataset(dsName, data=coords["z"][idxB], dtype=np.float64)
      pass

    # for block 3
    idxB = 2
    dsName = "Block-" + "%02d"%idxB + "-P"

    ista = numbPtsB1 + numbPtsB2; iend = numbPtsB1 + numbPtsB2 + numbPtsB3
    grp.create_dataset(dsName, data=output[ista:iend], dtype=np.float64)

    # 写入坐标
    if coords is not None:
      dsName = "Block-" + "%02d"%idxB + "-X"
      grp.create_dataset(dsName, data=coords["x"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Y"
      grp.create_dataset(dsName, data=coords["y"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Z"
      grp.create_dataset(dsName, data=coords["z"][idxB], dtype=np.float64)
      pass

    # for block 4
    idxB = 3
    dsName = "Block-" + "%02d"%idxB + "-P"

    ista = numbPtsB1 + numbPtsB2 + numbPtsB3
    iend = numbPtsB1 + numbPtsB2 + numbPtsB3 + numbPtsB4

    grp.create_dataset(dsName, data=output[ista:iend], dtype=np.float64)

    # 写入坐标
    if coords is not None:
      dsName = "Block-" + "%02d"%idxB + "-X"
      grp.create_dataset(dsName, data=coords["x"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Y"
      grp.create_dataset(dsName, data=coords["y"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Z"
      grp.create_dataset(dsName, data=coords["z"][idxB], dtype=np.float64)
      pass

    # for block 5
    idxB = 4
    dsName = "Block-" + "%02d"%idxB + "-P"

    ista = numbPtsB1 + numbPtsB2 + numbPtsB3 + numbPtsB4
    iend = numbPtsB1 + numbPtsB2 + numbPtsB3 + numbPtsB4 + numbPtsB5

    grp.create_dataset(dsName, data=output[ista:iend], dtype=np.float64)

    # 写入坐标
    if coords is not None:
      dsName = "Block-" + "%02d"%idxB + "-X"
      grp.create_dataset(dsName, data=coords["x"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Y"
      grp.create_dataset(dsName, data=coords["y"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Z"
      grp.create_dataset(dsName, data=coords["z"][idxB], dtype=np.float64)
      pass

    # for block 6
    idxB = 5
    dsName = "Block-" + "%02d"%idxB + "-P"

    ista = numbPtsB1 + numbPtsB2 + numbPtsB3 + numbPtsB4 + numbPtsB5
    iend = numbPtsB1 + numbPtsB2 + numbPtsB3 + numbPtsB4 + numbPtsB5 + numbPtsB6

    grp.create_dataset(dsName, data=output[ista:iend], dtype=np.float64)

    # 写入坐标
    if coords is not None:
      dsName = "Block-" + "%02d"%idxB + "-X"
      grp.create_dataset(dsName, data=coords["x"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Y"
      grp.create_dataset(dsName, data=coords["y"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Z"
      grp.create_dataset(dsName, data=coords["z"][idxB], dtype=np.float64)
      pass

    # for block 7
    idxB = 6
    dsName = "Block-" + "%02d"%idxB + "-P"

    ista = numbPtsB1 + numbPtsB2 + numbPtsB3 + numbPtsB4 + numbPtsB5 + numbPtsB6
    iend = numbPtsB1 + numbPtsB2 + numbPtsB3 + numbPtsB4 + numbPtsB5 + numbPtsB6 + numbPtsB7

    grp.create_dataset(dsName, data=output[ista:iend], dtype=np.float64)

    # 写入坐标
    if coords is not None:
      dsName = "Block-" + "%02d"%idxB + "-X"
      grp.create_dataset(dsName, data=coords["x"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Y"
      grp.create_dataset(dsName, data=coords["y"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Z"
      grp.create_dataset(dsName, data=coords["z"][idxB], dtype=np.float64)
      pass

    # for block 8
    idxB = 7
    dsName = "Block-" + "%02d"%idxB + "-P"

    ista = numbPtsB1 + numbPtsB2 + numbPtsB3 + numbPtsB4 + numbPtsB5 + numbPtsB6 + numbPtsB7
    iend = numbPtsB1 + numbPtsB2 + numbPtsB3 + numbPtsB4 + numbPtsB5 + numbPtsB6 + numbPtsB7 + numbPtsB8

    grp.create_dataset(dsName, data=output[ista:iend], dtype=np.float64)

    # 写入坐标
    if coords is not None:
      dsName = "Block-" + "%02d"%idxB + "-X"
      grp.create_dataset(dsName, data=coords["x"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Y"
      grp.create_dataset(dsName, data=coords["y"][idxB], dtype=np.float64)

      dsName = "Block-" + "%02d"%idxB + "-Z"
      grp.create_dataset(dsName, data=coords["z"][idxB], dtype=np.float64)
      pass
    pass

  # 打印损失函数
  def saveLossHistory2PNG(self):
    df = pandas.DataFrame(self.progress, columns=["Loss"])
    ax = df.plot( title  = "Loss history of Pressure",\
                  color  = "black",                   \
                  xlabel = "Epochs",                  \
                  ylabel = "Loss Value")
    ax.figure.savefig("lossHistory_P.png")
    pass
  pass

# 生成一个回归模型对象
R = Regression()

# train the model
epochs = 50

for i in range(epochs):
  print("Training Epoch", i+1, "of", epochs)

  for bc, label, _ in fsDataset_train:
    R.train(bc, label)
    pass
  pass

# 绘制损失函数历史
R.saveLossHistory2PNG()

# 预测

fsDataset_test = FSimDataset(filePathH5, listTestCase)
print(listTestCase)

#len_test = fsDataset_test.numCases
# for C025
inp, pField, coords = fsDataset_test[0]
print(fsDataset_test.caseList)
#print(type(inp), type(pField))

#outPresTorch = R.forward(inp)

#outPres = outPresTorch.detach().numpy()

#print(outPres[100])
#print(type(outPres))

R.write2HDF(inp, Path("./fnn.h5"),coords=coords)

