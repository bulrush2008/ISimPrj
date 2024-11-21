'''
This is the first model to predict flow field.
The model use the a fully-connected nural networks, also named as Fast-Foward neural Netorks, FNN.
The input is boundary condition, while the output is flow field.

@author Xia, Shuning
@date   2024.11.15
@aff    simpop.cn
'''

# import modules, esp torch, numpy and matplotlib
import torch as tch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas
import matplotlib.pyplot as plt

# the premodule
class MnistDataset(Dataset):
  def __init__(self, csv_file): ## should update
    self.data_df = pandas.read_csv(csv.file, header=none)
    pass

  def __len__(self):
    return len(self.data_df)

  def __getitem__(self, index):
    # image target (label).
    # Later, it would denotes the index of data of flow field
    label = self.data_df.iloc[index, 0]
    target = torch.zeros((10))
    target[label] = 1.0

    # image data, normalized from 0-255 to 0-1
    image_values = torch.FloatTensor(self.data_df.iloc[index, 1:].values)/255.0

    # return label, image tensor and target tensor
    return label, image_values, target

  def plot_image(self, index):
    img = self.data_df.iloc[index, 1:].values.reshape(28,28)
    plt.title("label = "+str(self.data_df.iloc[index, 0]))
    plt.imshow(img, interploation='none', cmap="Blues")
    pass
  pass


# def a class, defining the structure, optimiser, predict and input function inside it
class Classifier(nn.Module):
  # 初始化 PyTorch 的父类
  def __init__(self):
    super().__init__()

    # 定义神经网络层
    self.model = nn.Sequential(
      nn.linear(784, 200), # 784 must be adapted for FSim project: Boundary Condtion
      nn.Sigmoid(),
      nn.linear(200, 100),
      nn.Sigmoid(),
      nn.Linear(100, 100),
      nn.Sigmoid(),
      nn.Linear(100, 784) # 784 must be adapted for FSim Project: Flow Field
    )

    # 创建损失函数
    self.loss_function = nn.MSELoss()

    # 创建优化器
    self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
    pass

  def forward(self, inputs):
    # 直接运行模型，获得预测结果
    return self.model(inputs)

  def train(self, inputs, targets):
    '''
    inputs: boundary condition
    outputs: flow field
    ''' 
    outputs = self.forward(intputs)

    # 计算损失值
    loss = self.loss_function(outputs, targets)

    # increase counter and accumulate error every 10 (!every epoch better)
    self.counter += 1
    if(self.counter%10 == 0):
      self.progress.append(loss.item())
      pass
    if(self.counter%10000 == 0):
      print("counter = ", self.counter)
      pass

    # zero gradients, perform a backward pass, and update the weights
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  def plot_progress(self):
    df = pandas.DataFrame(self.progress, columns=['loss'])
    df.plot(ylim=(0,1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True)
    pass
  pass







# train


# postprocess