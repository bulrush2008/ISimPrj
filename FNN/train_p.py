
"""
The FNN model is used to prediction Pressure field. And also as a base code
for Other field variables.

@author     @data       @aff        @version
Xia, S      24.12.19    Simpop.cn   v2.x
"""

# --------------- import libraries ----------------------
import torch
import numpy as np

from pathlib import Path

from Common.idxList     import idxList, numOfAllCases
from Common.Regression  import Regression
from Common.FSimDataset import FSimDataset


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

fsDataset_train = FSimDataset(filePathH5, listTrainCase)

# ------------ 生成一个回归模型对象，并执行训练 ----------------
R = Regression()

# train the model
epochs = 1

for i in range(epochs):
  print("Training Epoch", i+1, "of", epochs)

  for bc, label, _ in fsDataset_train:
    #print(bc)
    R.train(bc, label)
    pass
  pass

# ---------- 训练完毕，绘制损失函数历史 ----------------------
DirPNG = Path("./Pics")
R.saveLossHistory2PNG(DirPNG)

# ------------- 预测，并于测试集比较 -------------------------
fsDataset_test = FSimDataset(filePathH5, listTestCase)
#print(listTestCase)

#len_test = fsDataset_test.numCases
# for C034
inp, pField, coords = fsDataset_test[0]
#print(fsDataset_test.caseList)
#print(type(inp), type(pField))
print(len(coords["x"]))
print(coords["x"])

#outPresTorch = R.forward(inp)

#outPres = outPresTorch.detach().numpy()

#print(outPres[100])
#print(type(outPres))

# -------------------- 将预测数据写道 HDF5 文件中 -------------------
R.write2HDF(inp, Path("./fnn.h5"),coords=coords)

