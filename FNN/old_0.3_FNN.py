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

import numpy as np
import matplotlib.pyplot as plt

# call the premodule


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





# train


# postprocess