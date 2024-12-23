
import torch
import torch.nn as nn
import math
import pandas
import h5py

import numpy as np

from pathlib import Path

# Regression class: Core of FNN
class Regression(nn.Module):
  # 初始化 PyTorch 父类
  def __init__(self, varName:str):
    super().__init__()

    if varName not in ["P", "U", "V", "W", "T"]:
      raise ValueError("Error: the Variable Name must be P/U/V/W/T.")

    self.varName = varName

    # 初次设置 3 个隐藏层
    self.model = nn.Sequential(
      nn.Linear(3, 100),  # 3 inputs
      nn.LeakyReLU(0.02),
      nn.LayerNorm(100),

      nn.Linear(100,300),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(300),

      nn.Linear(300,1000),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(1000),

      nn.Linear(1000,125557), # output field, 8 block
      nn.Identity()
    )

    # 初始化权重，目前使用 He Kaiming 方法
    self._initialize_weights()

    # 回归问题，需要使用 MSE
    self.loss_function = nn.MSELoss()
    self.optimiser = torch.optim.Adam(self.parameters(), lr=0.01)

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
      print(f"    - {self.counter} Cases Trained for {self.varName} ...")
      pass

    # 梯度归零，反向传播，更新学习参数
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  def write2HDF(self, inp:torch.FloatTensor, dirFileHDF:Path, coords:list=None):
    grpName = "FNN_Out" # 相当于原“A Case”

    h5 = h5py.File(dirFileHDF, 'a')

    if grpName in h5:
      grp = h5[grpName]
    else:
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
    dsName = "Block-" + "%02d"%idxB + "-" + self.varName

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
    dsName = "Block-" + "%02d"%idxB + "-" + self.varName

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
    dsName = "Block-" + "%02d"%idxB + "-" + self.varName

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
    dsName = "Block-" + "%02d"%idxB + "-" + self.varName

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
    dsName = "Block-" + "%02d"%idxB + "-" + self.varName

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
    dsName = "Block-" + "%02d"%idxB + "-" + self.varName

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
    dsName = "Block-" + "%02d"%idxB + "-" + self.varName

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
    dsName = "Block-" + "%02d"%idxB + "-" + self.varName

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
  def saveLossHistory2PNG(self, outDir:Path)->None:
    df = pandas.DataFrame(self.progress, columns=["Loss"])
    ax = df.plot( title  = f"Loss history of {self.varName}", \
                  color  = "black",             \
                  xlabel = "Epochs",            \
                  ylabel = "Loss Value",        \
                  logy   = True)
    outFile = outDir.joinpath(f"lossHistory_{self.varName}.png")
    ax.figure.savefig(outFile)
    pass
  pass