
"""
Train FNN Model

@author     @data       @aff        @version
Xia, S      2025.2.13   Simpop.cn   v6.x
"""
import sys
import json

import h5py
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

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
    cur_dir = Path(__file__).parent
    json_path = cur_dir.joinpath("FNN_Train.json")

    with open(json_path, 'r') as inp:
      data = json.load(inp)

    ratioTest = data["test_ratio"]  # e.g. 0.2
    caseSet = CaseSet(ratio=ratioTest)

    trnSet, tstSet = caseSet.splitSet()

    self.trnSet = trnSet
    self.tstSet = tstSet

    # path of data used as training and possibly test
    matrix_data_path = data["train_data"]

    cur_dir = Path(__file__).parent.parent
    self.filePathH5 = cur_dir.joinpath(matrix_data_path)

    self.fieldList = data["vars"]

    # data storing residuals between CFD field and prediction
    #   including both for train and test sets
    self.res_trn_hist = {}
    self.res_tst_hist = {}

    for var in self.fieldList.keys():
      self.res_trn_hist[var] = []
      self.res_tst_hist[var] = []

  def train(self):
  #-----------------------------------------------------------------------------
    """
    - train the fields one has assigned, which must be in ["P"/"T"/"U"/"V"/"W"]
    """

    fieldList = self.fieldList

    print(f"*Fields Models Will Be Trained with Epochs {fieldList}.")

    trnSet = self.trnSet
    tstSet = self.tstSet

    filePathH5 = self.filePathH5

    # directory of loss png
    cur_dir = Path(__file__).parent
    dirPNG = cur_dir.joinpath("Pics")
    if not dirPNG.exists(): dirPNG.mkdir(parents=True)

    # directory of model
    dirModel = cur_dir.joinpath("StateDicts")
    if not dirModel.exists(): dirModel.mkdir(parents=True)

    # train
    self._train(varList  = fieldList,
                trainSet = trnSet,
                testSet  = tstSet,
                dataPath = filePathH5,
                dirPNG   = dirPNG,
                dirModel = dirModel)

  def _train( self,
              varList  :dict,
              trainSet :list,
              testSet  :list,
              dataPath :Path,
              dirPNG   :Path,
              dirModel :Path )->None:
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
      cur_dir = Path(__file__).parent
      var_dict_path = cur_dir.joinpath(f"StateDicts/dict_{var}.pth")

      if not var_dict_path.exists():
        var_dict_path = None
        print(f"Train from ZERO for {var}")
      else:
        print(f"Train from dict_{var}.pth")

      R = Regression(var, var_dict_path)

      print(f"*Now we are training {var} field:")

      # train the model
      epochs = varList[var]

      for i in range(epochs):
        print(f" >> Training {var}, epoch {i+1}/{epochs}")
        for inp, label, _ in fsDataset_train:
          R.train(inp, label)

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

        self.res_trn_hist[var].append(e_train)
        self.res_tst_hist[var].append(e_test)
        pass

      # write residuals for this "var"
      self.write_e_hists(var)

      # plot loss history and save
      R.saveLossHistory2PNG(dirPNG)

      ipic = 0
      for inp, field, _ in fsDataset_test:
        R.save_regression_png(order=ipic, inp=inp, target=field)
        ipic += 1

      # save model parameters
      model_dicts_name = dirModel.joinpath(f"dict_{var}.pth")
      torch.save(R.model.state_dict(), model_dicts_name)
    # now all variable models have been trained

  def write_e_hists(self, var:str):
  #-----------------------------------------------------------------------------
    """
    save the e_hist into a png file

    - var : string 变量，用于命名
    """
    fig, ax = plt.subplots(1,1)

    y1 = self.res_trn_hist[var]
    y2 = self.res_tst_hist[var]

    x = list(range(1,len(y1)+1))

    ax.plot(x, y1, label="Train")
    ax.plot(x, y2, label="Test")

    # log-y
    ax.set_yscale("log")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Linf norm")

    ax.legend()

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

    cur_dir = Path(__file__).parent
    fig.savefig(cur_dir.joinpath(f"Pics/resLinf_{var}-{current_time}.png"), dpi=200)
  # end class