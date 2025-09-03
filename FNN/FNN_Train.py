
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
    self.train_info: including variable and training epochs
    self.train_residuals
    self.test_residuals

    # 2 methods:
    self.train_loop()
    self.write_e_hists()
    """

    #== 读入 .json 设置文件
    cur_dir = Path(__file__).parent
    json_path = cur_dir.joinpath("FNN_Train.json")

    with open(json_path, 'r') as inp:
      data = json.load(inp)

    #== 读入算例列表，并分割
    ratioTest = data["test_ratio"]  # e.g. 0.2
    caseSet = CaseSet(ratio=ratioTest)

    self.train_set, self.test_set = caseSet.splitSet()

    # 记录数据集位置
    # path of data used as training and possibly test
    matrix_data_path = data["train_data"]

    cur_dir = Path(__file__).parent.parent
    self.h5file_path = cur_dir.joinpath(matrix_data_path)

    #记录训练信息，包含带训练的模型名称和训练周期数
    self.train_info = data["vars"]

    # 定义存储误差记录的列表变量
    self.train_residuals = {}
    self.test_residuals = {}

    for var in self.train_info.keys():
      self.train_residuals[var] = []
      self.test_residuals[var] = []

    # 检查图像存储目录和模型参数目录是否已经创建，如果不存在，则创建该目录
    cur_dir = Path(__file__).parent

    pic_dir = cur_dir.joinpath("Pics")
    if not pic_dir.exists(): pic_dir.mkdir(parents=True)

    model_dir = cur_dir.joinpath("StateDicts")
    if not model_dir.exists(): model_dir.mkdir(parents=True)

    print(f"> We will train {self.train_info}\n")

    self.fsDataset_train = {}
    self.fsDataset_test = {}

    self.regressions = {}
    # 初始化应用对象，类似于先创建全局变量
    for var in self.train_info.keys():
      self.fsDataset_train[var] = FSimDataset(self.h5file_path, self.train_set, var)
      self.fsDataset_test[var]  = FSimDataset(self.h5file_path, self.test_set,  var)

      var_dict_path = cur_dir.joinpath(f"StateDicts/dict_{var}.pth")

      if not var_dict_path.exists():
        var_dict_path = None
        print(f"> Train {var} from ZERO")
      else:
        print(f"> Train {var} from dict_{var}.pth")

      self.regressions[var] = Regression(var, var_dict_path)
    # 结束for-loop

    # 迭代计数器
    self.istep = 0
  # 结束 __init__

  def train_loop(self, var:str, numb:int) -> tuple[int, int]:
    """
    主训练循环，

    - 对变量 "var" 做训练迭代，迭代次数为 "numb"
    - 每次检查是否超过最大迭代次数(self.train_info['var'])，以及剩余迭代次数
      * 如果剩余次数小于 "numb"，则按照剩余次数进行训练
      * 如果剩余次数大于 "numb"，则迭代次数为 "numb"
    """

    # train the model
    epoch = self.train_info[var]
    real_numb = min(epoch - self.istep, numb)

    for i in range(real_numb):
      print(f"> {var}: epoch {self.istep+1}/{epoch}")

      for inp, label, _ in self.fsDataset_train[var]:
        self.regressions[var].train(inp, label)

      self.istep += 1

      # calculate and record residuals
      e_train = 0.0
      for inp, field, _ in self.fsDataset_train[var]:
        e_train = max(e_train, self.regressions[var].calc_Field_MSE(inp, field))

      e_test = 0.0
      for inp, field, _ in self.fsDataset_test[var]:
        e_test = max(e_test, self.regressions[var].calc_Field_MSE(inp, field))

      self.train_residuals[var].append(e_train)
      self.test_residuals[var].append(e_test)
    # 完成了此模型的此次阶段训练任务：训练次数=min{numb, epoch-istep}

    # plot residuals of "var"
    if self.istep >= epoch:
      print("")
      print(f"> Plot {var} error history")
      self.write_e_hists(var)

    # plot loss history
    if self.istep >= epoch:
      print(f"> Plot {var} loss history")

      cur_dir = Path(__file__).parent
      pic_dir = cur_dir.joinpath("Pics")
      self.regressions[var].saveLossHistory2PNG(pic_dir)

    # plot regression graph
    if self.istep >= epoch:
      print(f"> Plot {var} regression")

      ipic = 0
      for inp, field, _ in self.fsDataset_test[var]:
        self.regressions[var].save_regression_png(order=ipic, inp=inp, target=field)
        ipic += 1

    # save model parameters
    cur_dir = Path(__file__).parent
    model_dir = cur_dir.joinpath("StateDicts")
    model_dicts_name = model_dir.joinpath(f"dict_{var}.pth")
    torch.save(self.regressions[var].model.state_dict(), model_dicts_name)
    # 完成所有模型的训练

    return (self.istep, self.train_info[var])
  # 结束训练过程：train_loop

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

    cur_dir = Path(__file__).parent

    #current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    #fig.savefig(cur_dir.joinpath(f"Pics/resLinf_{var}-{current_time}.png"), dpi=200)

    # 图片名称不再显示时间戳
    fig.savefig(cur_dir.joinpath(f"Pics/residual_{var}.png"), dpi=100)
  # 结束函数：write_e_hists
# end class