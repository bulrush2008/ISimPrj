import torch
import torch.nn as nn
import math
import pandas
import h5py
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

class ModelPinn(nn.Module):
#===============================================================================
  """
  - ModelPinn class: Core of FnnPinn
  - 方法类
  """
  # initialize PyTorch pararent class
  def __init__(self, config:dict):
  #-----------------------------------------------------------------------------
    super().__init__()
    self.load_dict = config.get("load_dict", False) # 是否载入之前训练的模型
    self.dropout = config.get("dropout", None)
    self.learning_rate = config.get("learning_rate", 0.005)
    self.activation = config.get("activation", "leaky_relu")
    if torch.cuda.is_available():
        self.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        self.device = torch.device("mps")
    else:
        self.device = torch.device("cpu")
    
    self.dict_path = Path(config["dict_dir_path"]).joinpath(f"dict_{config['dict_load_name']}")

    print(f"*Use device: {self.device}")
    print(f"*Use dropout: {self.dropout}")
    print(f"*Use learning rate: {self.learning_rate}")
    print(f"*Use activation: {self.activation}")
    print(f"*Load Previous Model: {'yes' if self.load_dict else 'no'}")
    
    # Additional GPU information
    if torch.cuda.is_available():
        print(f"*CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"*CUDA Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Build model from configurable architecture
    self.model = self._build_model(config["architecture"])

    print("*Model structure:")
    print(self.model)
    
    self.model = self.model.to(self.device)

    if self.load_dict:
      self.model.load_state_dict(torch.load(self.dict_path))
      print(f"Load model from {self.dict_path}")
    else:
      # initialize weights，using He Kaiming method now
      self._initialize_weights()
      print(f"Initialize model weights")
      pass

    # regressive problem, MSE is proper
    self.loss_function = nn.MSELoss()
    self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # counter: record the trained times
    self.counter = 0
    self.progress = []  # loss of each train process

    # error estimation of train and test sets
    self.err_train = [] # float list
    self.err_test  = [] # float list
    pass

  def _build_model(self, architecture):
  #-----------------------------------------------------------------------------
    """
    Build neural network model from architecture configuration
    - architecture: list of [input_size, output_size] pairs
    """
    layers = []
    
    for i, (in_features, out_features) in enumerate(architecture):
        layers.append(nn.Linear(in_features, out_features))
        
        if i < len(architecture) - 1:
            if self.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.02))
            elif self.activation == "relu":
                layers.append(nn.ReLU())
            elif self.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.LeakyReLU(0.02))
                
            if self.dropout is not None and self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))

            layers.append(nn.LayerNorm(out_features))
    
    return nn.Sequential(*layers)

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
    # inputs = inputs.to(self.device)
    return self.model(inputs)

  def train(self, params, coords, targets):
  #-----------------------------------------------------------------------------
    """
    - 神经网络，根据输入和标签，进行训练

    - params : 控制参数
    - coords : 坐标
    - targets: 目标值
    """
    params = params.to(self.device)
    coords = coords.to(self.device)
    targets = targets.to(self.device)
    
    inputs = torch.cat([params, coords], dim=1) # (N, 6)
    outputs = self.forward(inputs) # (N, 1)
    loss = self.loss_function(outputs, targets)
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    return loss.item()

#   def write2HDF(self, inp:torch.FloatTensor, dirFileHDF:Path, coords:list=None):
#   #-----------------------------------------------------------------------------
#     """
#     - 将预测数据，写入 HDF 数据库
#     - 如有必要，会写入坐标
#     """
#     # h5 文件已经在外部打开，这里只需要创建一个组，用来管理模型的预测数据即可
#     grpName = "FNN_Out" # a case data is a group

#     # 以附加的方式打开
#     h5 = h5py.File(dirFileHDF, 'a')

#     # 不知道某变量流场是否已经写入，需要检测、区分
#     if grpName in h5:
#       grp = h5[grpName] # if the group existed
#     else:
#       grp = h5.create_group(grpName)  # if not, created it
#       pass

#     # 根据参数化输入，预测流场
#     # the predicted data should be detached and converted to numpy format
#     output = self.forward(inp).detach().cpu().numpy()

#     # write data into h5 database directly
#     dsName = f"{self.varName}"
#     grp.create_dataset(dsName, data=output, dtype=np.float64)

#     # write coordinates it necessary
#     if coords is not None:
#       dsName = "Coords-X"
#       grp.create_dataset(dsName, data=coords["x"], dtype=np.float64)

#       dsName = "Coords-Y"
#       grp.create_dataset(dsName, data=coords["y"], dtype=np.float64)

#       dsName = "Coords-Z"
#       grp.create_dataset(dsName, data=coords["z"], dtype=np.float64)
#       pass

#     h5.close()
#     pass

#   def saveLossHistory2PNG(self, outDir:Path)->None:
#   #-----------------------------------------------------------------------------
#     """
#     打印损失函数

#     - outDir: 损失函数打印图片的输出目录
#     """
#     df = pandas.DataFrame(self.progress, columns=["Loss"])
#     ax = df.plot( title  = f"Loss history of {self.varName}",
#                   color  = "black",
#                   xlabel = "Number of Trained Cases",
#                   ylabel = "Loss Value",
#                   logy   = True)

#     var = self.varName
#     current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

#     outFile = outDir.joinpath(f"lossHist_{var}-{current_time}.png")
#     ax.figure.savefig(outFile)
#     pass  # end funcsaveLossHistory2PNG

  def field_error(self, params, coords, targets, error_type="L-inf"):
  #-----------------------------------------------------------------------------
    """
    Calculate the error of each case between prediction and real.
    Function's three input parameters are same with 'self.train(...)'

    - params: input parameters
    - coords: coordinates
    - targets: data of the real field of each case
    """
    # Move inputs and targets to device
    params = params.to(self.device)
    coords = coords.to(self.device)
    targets = targets.to(self.device)

    # Set the model to eval mode so that dropout and batch norm behave correctly
    self.model.eval()

    inputs = torch.cat([params, coords], dim=1) # (N, 6)
    output = self.forward(inputs).detach().cpu().numpy()

    # a/b both of type torch.FloatTensor
    a = output; b = targets.detach().cpu().numpy()
    #e = sum(abs(a-b))
    if error_type == "L-inf":
      e = np.max(np.abs(a-b), axis=1)
    elif error_type == "L-2":
      e = np.sqrt(np.mean((a-b)**2, axis=1))
    else:
      raise ValueError(f"Error: {error_type} error not implemented.")
    return e

#   def save_regression_png(self, order, inp, target):
#   #-----------------------------------------------------------------------------
#     """
#     绘制回归图，每个图点的
#     - 横坐标: CFD 仿真结果
#     - 纵坐标: 代理模型预测结果

#     - order: 测试算例的标号索引
#     - inp: 对应算例的模型输入参数
#     - target: 算例对应的流场，它是模型的输入目标
#     """
#     x = target.detach().cpu().numpy()
#     y = self.forward(inp).detach().cpu().numpy()

#     imax = max(max(x),max(y))
#     imin = min(min(x),min(y))

#     irange = imax - imin

#     x = (x-imin) / irange
#     y = (y-imin) / irange
    
#     fig, ax = plt.subplots(1,1)
#     ax.plot(x,y,  ls='',
#                   marker='o',
#                   markersize=2,
#                   markerfacecolor='black',
#                   markeredgecolor="black",
#                   label="Regression")
#     ax.plot([0,1], [0,1], c='orange')

#     ax.set_xlabel("CFD")
#     ax.set_ylabel("ML Eval")

#     ax.set_title(f"Regression for variable {self.varName}")

#     ax.legend()

#     var = self.varName
#     current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

#     fig.savefig(f"./Pics/regression_{var}-{order:03d}-{current_time}.png")
#     plt.close()
#     pass
#   pass  # end class ModelPinn

if __name__=="__main__":
#===============================================================================
  R = ModelPinn({"architecture": [[3, 100], [100, 200], [200, 3]]}, "T")
  pass