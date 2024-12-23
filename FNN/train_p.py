
"""
This function serves training the FNN model of pressure field.

@author     @data       @aff        @version
Xia, S      24.12.19    Simpop.cn   v3.x
"""

# --------------- import libraries ----------------------
import torch
import numpy as np

from pathlib import Path

from Common.idxList     import idxList, numOfAllCases
from Common.Regression  import Regression
from Common.FSimDataset import FSimDataset

def train_p(numOfEpochs:int=5)->bool:
  iSuccess = False

  varName = "P" # for pressure field

  # ----------------- 分割并确定训练数据表、测试数据表 ----------------------
  # split the data, 49 = 40 + 9
  ratioTest = 0.2

  sizeOfTestSet = np.int64(numOfAllCases * ratioTest)

  # 42 是随机种子，其它整数也可以
  np.random.seed(42)
  permut = np.random.permutation(numOfAllCases)

  listTestCase = []
  for i in permut[:sizeOfTestSet]:
    theCase = "C" + "%03d"%idxList[i]
    listTestCase.append(theCase)

  listTrainCase = []
  for i in permut[sizeOfTestSet:]:
    theCase = "C" + "%03d"%idxList[i]
    listTrainCase.append(theCase)

  # ------------------ 数据类的初始化 ---------------------
  filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")
  #aLive = filePathH5.exists()
  #print(aLive)

  fsDataset_train = FSimDataset(filePathH5, listTrainCase, varName)

  # ------------ 生成一个回归模型对象，并执行训练 ----------------
  R = Regression(varName)

  # train the model
  epochs = numOfEpochs
  for i in range(epochs):
    print("Training Epoch", i+1, "of", epochs)
    for inp, label, _ in fsDataset_train:
      R.train(inp, label)
      pass
    pass

  # ---------- 训练完毕，绘制损失函数历史 ----------------------
  DirPNG = Path("./Pics")
  R.saveLossHistory2PNG(DirPNG)

  # ------------- 预测，并与测试集比较 -------------------------
  fsDataset_test = FSimDataset(filePathH5, listTestCase, varName)

  # for C034
  inp, _, coords = fsDataset_test[0]

  # 1-先预测数据，2-再写到 HDF5 文件中
  # 坐标是可选输入数据
  R.write2HDF(inp, Path("./fnn.h5"), coords=coords)

  iSuccess = True
  return iSuccess

