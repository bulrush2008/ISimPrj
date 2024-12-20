
import torch
import numpy as np

from pathlib import Path

from Common.Regression import Regression
from Common.Discriminator import Discriminator
from Common.idxList import idxList, numOfAllCases

def train_p(numOfEpochs:int=10)->bool:
  iSuccess = False

  varName = "P" # now for pressure var

  # ----------------- 分割并确定训练数据表、测试数据表 ----------------------
  # split the data, 49 = 40 + 9
  ratioTest = 0.2

  sizeOfTestSet = np.int64(numOfAllCases * ratioTest)

  # 42 是随机种子，其它整数也可以
  np.random.seed(42)
  permut = np.random.permutation(numOfAllCases)

  # training set
  listTestCase = []
  for i in permut[:sizeOfTestSet]:
    theCase = "C" + "%03d"%idxList[i]
    listTestCase.append(theCase)

  # test set
  listTrainCase = []
  for i in permut[sizeOfTestSet:]:
    theCase = "C" + "%03d"%idxList[i]
    listTrainCase.append(theCase)

  # ------------------ 数据类的初始化 ---------------------
  filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")
  #aLive = filePathH5.exists()
  #print(aLive)

  fsDataset_train = FSimDataset(filePathH5, listTrainCase, varName)

  # ------------ 声明生成器、鉴别器对象 ----------------

  # Now for pressure only
  G = Regression("P")
  D = Discriminator()

  for epoch in range(numOfEpochs):
    print(f"Training Epoch {epoch+1} of {numOfEpochs}")

    # train Discriminator and Generator
    for bc, field, _ in fsDataset_train:
      # train Discriminator on <True>
      D.train(field, torch.FloatTensor([1.0], dtype=torch.Float64))

      # train Discriminator on <False>
      D.train(G.forward(bc).detach(), torch.FloatTensor([0.0], dtype=torch.Float64))

      # train Generator
      G.train(D, bc, torch.FloatTensor([1.0]))
      pass
    pass

  iSuccess = True
  return iSuccess
  pass