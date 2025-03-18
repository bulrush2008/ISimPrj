
import torch
import torch.nn as nn
import math
import pandas
import h5py

import numpy as np

from pathlib import Path

class Regression(nn.Module):
#===============================================================================
  """
  - Regression class: Core of FNN
  - 方法类
  """
  # initialize PyTorch pararent class
  def __init__(self, varName:str, dictPath:Path=None):
  #-----------------------------------------------------------------------------
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

    if dictPath is not None:
      self.model.load_state_dict(torch.load(dictPath))
    else:
      # initialize weights，using He Kaiming method now
      self._initialize_weights()
      pass

    # regressive problem, MSE is proper
    self.loss_function = nn.MSELoss()
    self.optimiser = torch.optim.Adam(self.parameters(), lr=0.01)

    # counter: record the trained times
    self.counter = 0
    self.progress = []  # loss of each train process

    # error estimation of train and test sets
    self.err_train = [] # float list
    self.err_test  = [] # float list
    pass

  def _initialize_weights(self):
  #-----------------------------------------------------------------------------
    """
    - inner function, call only once at initialization
    - configure initial model weights
    """
    for m in self.model:
      if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
          nn.init.zeros_(m.bias)
          pass
        pass
      pass
    pass

  def forward(self, inputs):
  #-----------------------------------------------------------------------------
    return self.model(inputs)

  def train(self, inputs, targets):
  #-----------------------------------------------------------------------------
    """
    - 神经网络，根据输入和标签，进行训练

    - inputs : 神经网络的输入
    - targets: 神经网络的教师标签
    """
    outputs = self.forward(inputs)

    # calculate loss
    loss = self.loss_function(outputs, targets)

    # each train step, the loss must be added
    self.progress.append(loss.item())

    self.counter += 1
    if(self.counter%10 == 0):  # print training info onto screen every 10 cases
      print(f"    - {self.counter} Cases Trained ...")
      pass

    # grad must set back to zero
    # back propagation, and update the learning rate parameter
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  def write2HDF(self, inp:torch.FloatTensor, dirFileHDF:Path, coords:list=None):
  #-----------------------------------------------------------------------------
    """
    - 将预测数据，写入 HDF 数据库
    - 如有必要，会写入坐标
    """
    # h5 文件已经在外部打开，这里只需要创建一个组，用来管理模型的预测数据即可
    grpName = "FNN_Out" # a case data is a group

    # 以附加的方式打开
    h5 = h5py.File(dirFileHDF, 'a')

    # 不知道某变量流场是否已经写入，需要检测、区分
    if grpName in h5:
      grp = h5[grpName] # if the group existed
    else:
      grp = h5.create_group(grpName)  # if not, created it
      pass

    # 根据参数化输入，预测流场
    # the predicted data should be detached and converted to numpy format
    output = self.forward(inp).detach().numpy()

    # write data into h5 database directly
    dsName = f"{self.varName}"
    grp.create_dataset(dsName, data=output, dtype=np.float64)

    # write coordinates it necessary
    if coords is not None:
      dsName = "Coords-X"
      grp.create_dataset(dsName, data=coords["x"], dtype=np.float64)

      dsName = "Coords-Y"
      grp.create_dataset(dsName, data=coords["y"], dtype=np.float64)

      dsName = "Coords-Z"
      grp.create_dataset(dsName, data=coords["z"], dtype=np.float64)
      pass

    h5.close()
    pass

  def saveLossHistory2PNG(self, outDir:Path)->None:
  #-----------------------------------------------------------------------------
    """
    - 打印损失函数
    """
    df = pandas.DataFrame(self.progress, columns=["Loss"])
    ax = df.plot( title  = f"Loss history of {self.varName}",
                  color  = "black",
                  xlabel = "Number of Trained Cases",
                  ylabel = "Loss Value",
                  logy   = True)
    outFile = outDir.joinpath(f"lossHistory_{self.varName}.png")
    ax.figure.savefig(outFile)
    pass  # end funcsaveLossHistory2PNG

  def calc_dataset_error(self, dataset:list):
  #-----------------------------------------------------------------------------
    """
    Calculate the errors either for train or test data set.add

    - dataset: list of strings, containing the cases name of train/test dataset
    """
    print(self.varName)
    print(dataset)
    pass

  def print_error_func(self):
  #-----------------------------------------------------------------------------
    pass  # end func print_error_func
  pass  # end class Regression

if __name__=="__main__":
#===============================================================================
  R = Regression("T")

  R.calc_dataset_error(['c1', 'case2'])
  pass