
"""
Train FNN Model

@author     @data       @aff        @version
Xia, S      2025.8.28   Simpop.cn   v6.x
"""

# standard libs
import sys
import json

from pathlib import Path
from datetime import datetime

# third-party libs
import h5py
import torch
import matplotlib.pyplot as plt

# self-defined libs
from Common.CaseSet import CaseSet
from Common.FSimDataset import FSimDataset
from Common.Regression  import Regression


class FNN_Train(object):
  """
  一个模型对应一个流场，比如温度场 T，对应一个NN模型

  - 实例化网络对象
  - 实例化数据对象
  - 组织训练过程
  """
  def __init__(self):
    """
    # 6 attributes:
    self.train_set
    self.test_set
    self.h5file_path
    self.model_list
    self.train_residuals
    self.test_residuals

    # 2 methods:
    self.train_loop()
    self.write_e_hists()
    """

    # split the cases into train and test sets
    # now: 125 = 100 + 25
    cur_dir = Path(__file__).parent
    json_path = cur_dir.joinpath("FNN_Train.json")

    with open(json_path, 'r') as inp:
      data = json.load(inp)

    ratioTest = data["test_ratio"]  # e.g. 0.2
    caseSet = CaseSet(ratio=ratioTest)

    self.train_set, self.test_set = caseSet.splitSet()

    # path of data used as training and possibly test
    matrix_data_path = data["train_data"]

    cur_dir = Path(__file__).parent.parent
    self.h5file_path = cur_dir.joinpath(matrix_data_path)

    self.model_list = data["vars"]

    # data storing residuals between CFD field and prediction
    #   including both for train and test sets
    self.train_residuals = {}
    self.test_residuals = {}

    for var in self.model_list.keys():
      self.train_residuals[var] = []
      self.test_residuals[var] = []

  def train_loop(self):
    """
    - train the fields one has assigned, which must be in ["P"/"T"/"U"/"V"/"W"]
    """

    fields = list(self.model_list.keys())
    epochs = list(self.model_list.values())
    print(f"> Models {fields} trained with epochs {epochs}.")

    # directory of loss png
    cur_dir = Path(__file__).parent
    pic_dir = cur_dir.joinpath("Pics")
    if not pic_dir.exists(): pic_dir.mkdir(parents=True)

    # directory of model
    model_dir = cur_dir.joinpath("StateDicts")
    if not model_dir.exists(): model_dir.mkdir(parents=True)

    # extract the var names
    fields = []
    for key in self.model_list.keys():
      fields.append(key)

    # including all trained models
    #models = {}

    # train fields
    for var in fields:
      # obj to get the train data set
      # train set serves as (1) train & (2) error estimation
      fsDataset_train = FSimDataset(self.h5file_path, self.train_set, var)

      # obj to get the test data set
      # test set servers only as erro estimation
      fsDataset_test = FSimDataset(self.h5file_path, self.test_set, var)

      # gen a obj as regression, and then train the model
      cur_dir = Path(__file__).parent
      var_dict_path = cur_dir.joinpath(f"StateDicts/dict_{var}.pth")

      print("") # 打印空行，隔开两个模型的训练过程
      if not var_dict_path.exists():
        var_dict_path = None
        print(f"> Train {var} from ZERO")
      else:
        print(f"> Train {var} from dict_{var}.pth")

      R = Regression(var, var_dict_path)

      #print(f"> Start training {var} field:")

      # train the model
      epochs = self.model_list[var]

      for i in range(epochs):
        print(f"> {var}: epoch {i+1}/{epochs}")
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

        self.train_residuals[var].append(e_train)
        self.test_residuals[var].append(e_test)
        pass

      # write residuals for this "var"
      print(f"> Plott {var} error history")
      self.write_e_hists(var)

      # plot loss history and save
      print(f"> Plott {var} loss history")
      R.saveLossHistory2PNG(pic_dir)

      print(f"> Plott {var} regression")
      ipic = 0
      for inp, field, _ in fsDataset_test:
        R.save_regression_png(order=ipic, inp=inp, target=field)
        ipic += 1

      # save model parameters
      model_dicts_name = model_dir.joinpath(f"dict_{var}.pth")
      torch.save(R.model.state_dict(), model_dicts_name)
    # now all variable models have been trained

  def write_e_hists(self, var:str):
    """
    plot error history to pics

    - var : string 变量，用于命名
    """

    fig, ax = plt.subplots(1,1)

    y1 = self.train_residuals[var]
    y2 = self.test_residuals[var]

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