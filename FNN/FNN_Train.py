
"""
This is main function to call:
  - split the data into train and test sets
  - train,
  - predict, and save into the database: .h5

@author     @data       @aff        @version
Xia, S      2025.2.13   Simpop.cn   v6.x
"""
import sys
import json

import h5py
import torch
import matplotlib.pyplot as plt

from pathlib import Path

from Common.CaseSet import CaseSet
from Common.FSimDataset import FSimDataset
from Common.Regression  import Regression

class FNN_Train(object):
#===============================================================================
  """
  - 应用任务类

  - 调用方法类和数据类，实现特定的应用任务
  """
  def __init__( self ):
  #-----------------------------------------------------------------------------
    # split the cases into train and test sets
    # now: 125 = 100 + 25
    with open("./FNN_Train.json", 'r') as inp:
      data = json.load(inp)
      pass

    ratioTest = data["test_ratio"]  # e.g. 0.2
    caseSet = CaseSet(ratio=ratioTest)

    trnSet, tstSet = caseSet.splitSet()

    self.trnSet = trnSet
    self.tstSet = tstSet

    # path of data used as training and possibly test
    matrix_data_path = data["train_data"]
    self.filePathH5 = Path(matrix_data_path)

    self.fieldList = data["vars"]

    self.e_hist = {"train":[], "test":[]}
    pass

  def train(self):
  #-----------------------------------------------------------------------------
    # train the fields one has assigned, which must be in
    # ["P", "T", "U", "V", "W"]
    # the order in list does not matter
    fieldList = self.fieldList

    print(f"*Fields Models Will Be Trained with Epochs {fieldList}.")

    trnSet = self.trnSet
    tstSet = self.tstSet

    filePathH5 = self.filePathH5

    models = self._train( varList  = fieldList,
                          trainSet = trnSet,
                          testSet  = tstSet,
                          dataPath = filePathH5 )

    dirPNG = Path("./Pics")
    if not dirPNG.exists(): dirPNG.mkdir(parents=True)

    dirModel = Path("./StateDicts")
    if not dirModel.exists(): dirModel.mkdir(parents=True)

    for var in fieldList.keys():
      #------------------------------------------------------------------------
      # plot loss history and save
      models[var].saveLossHistory2PNG(dirPNG)

      #------------------------------------------------------------------------
      # save model parameters
      model_dicts_name = dirModel.joinpath(f"dict_{var}.pth")
      torch.save(models[var].model.state_dict(), model_dicts_name)
      pass
    pass

  def _train( self,
              varList :dict,
              trainSet:list,
              testSet :list,
              dataPath:Path )->dict:
  #-----------------------------------------------------------------------------
    """
    Train the FNN model by a give trainset, in which some cases field included.
    - varList : dict of epochs for each field, such as ["P":1,"T":2]
    - trainSet: list of case names in train set, each is a string
    - testSet : list of case names in test set, each is a string
    - dataPath: path of data of train set
    """

    # extract the var names
    fields = []
    for key in varList.keys():
      fields.append(key)
      pass
  
    # including all trained models
    models = {}

    # train fields
    for var in fields:
      # obj to get the train data set
      # train set serves as (1) train & (2) error estimation
      fsDataset_train = FSimDataset(dataPath, trainSet, var)

      # obj to get the test data set
      # test set servers only as erro estimation
      fsDataset_test = FSimDataset(dataPath, testSet, var)

      # gen a obj as regression, and then train the model
      var_dict_path = Path(f"./StateDicts/dict_{var}.pth")

      if not var_dict_path.exists():
        var_dict_path = None
        print(f"Train from ZERO for {var}")
      else:
        print(f"Train from dict_{var}.pth")
        pass

      R = Regression(var, var_dict_path)

      print(f"*Now we are training {var} field:")

      # train the model
      epochs = varList[var]

      for i in range(epochs):
        print(f"  > Training Epoch {i+1} of {epochs} for {var}")
        for inp, label, _ in fsDataset_train:
          R.train(inp, label)
          pass

        # we need calculate field error to do estimation for both train and
        #   test data set

        # for the train set
        e_train = 0.0
        for inp, field, _ in fsDataset_train:
          e_train = max(e_train, R.calc_Field_MSE(inp, field))
          pass

        # for the test set
        e_test = 0.0
        for inp, field, _ in fsDataset_test:
          e_test = max(e_test, R.calc_Field_MSE(inp, field))
          pass

        self.e_hist["train"].append(e_train)
        self.e_hist["test"].append(e_test)
        pass

      models[var] = R
      pass  # end all var-models training

    # now all variable models have been trained
    return models

  def write_e_hists(self, time:str):
  #-----------------------------------------------------------------------------
    """
    save the e_hist into a png file

    - time: string 变量，充当本次打印图片的时间戳
    """
    fig, ax = plt.subplots(1,1)

    y1 = self.e_hist["train"]
    y2 = self.e_hist["test"]

    x = list(range(1,len(y1)+1))

    ax.plot(x, y1)
    ax.plot(x, y2)

    # log-y
    ax.set_yscale("log")

    fig.savefig(f"./Pics/errorsL1-{time}.png")
    pass
  pass  # end class