
"""
The FNN model is used to prediction Pressure field. And also as a base code
for Other field variables.

@author     @data       @aff        @version
Xia, S      24.12.10    Simpop.cn   v0.1
"""

import torch
import torch.nn as nn
import pandas

class Regression(nn.Module):
  # 初始化 PyTorch 父类
  def __init__(self):
    super().__init__()

    # 初次设置 3 个隐藏层
    self.model = nn.Sequential(
      nn.linear(3, 100),  # 3 inputs
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
    if(self.count%1 == 0):  # 对每个算例数据，记录损失值
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
    df = pandas.DataFrame(self.progress, columns=["Loss"])
    df.plot()
    pass
  pass
