
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
