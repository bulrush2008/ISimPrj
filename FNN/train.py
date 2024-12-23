
"""
Train all fields by FNN model

@author     @data       @aff        @version
Xia, S      24.12.23    Simpop.cn   v3.x
"""

# -----------------------------------------------------------------------------
import torch

from pathlib import Path

from Common.Regression  import Regression
from Common.FSimDataset import FSimDataset

# -----------------------------------------------------------------------------
def train(numOfEpochs:int=5, fields:list=["T"], trainSet:list=["C001"] )->bool:
  """
  Train the FNN model by a give trainset, in which some cases field included.
  """
  iSuccess = False

  #----------------------------------------------------------------------------
  # 数据类的初始化
  filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")
  #aLive = filePathH5.exists()
  #print(aLive)

  fsDataset_train = FSimDataset(filePathH5, trainSet, fields[0])

  #----------------------------------------------------------------------------
  # 生成一个回归模型对象，并执行训练
  R = Regression(fields[0])

  # train the model
  epochs = numOfEpochs
  for i in range(epochs):
    print(f"Training Epoch {i+1} of {epochs} for {fields[0]}")
    for inp, label, _ in fsDataset_train:
      R.train(inp, label)
      pass
    pass

  #----------------------------------------------------------------------------
  # 训练完毕，绘制损失函数历史
  DirPNG = Path("./Pics")
  R.saveLossHistory2PNG(DirPNG)

  # ------------- 预测，并与测试集比较 -------------------------
  #fsDataset_test = FSimDataset(filePathH5, listTestCase, varName)

  # for C034
  #inp, _, coords = fsDataset_test[0]

  # 1-先预测数据，2-再写到 HDF5 文件中
  # 坐标是可选输入数据
  #R.write2HDF(inp, Path("./fnn.h5"), coords=coords)

  iSuccess = True
  return iSuccess

