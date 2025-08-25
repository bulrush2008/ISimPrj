
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

from pathlib import Path

from Common.CaseSet import CaseSet
from Common.FSimDataset import FSimDataset
from Common.Regression  import Regression

class FNN_Eval(object):
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
    with open(cur_dir.joinpath("FNN_Eval.json"), 'r') as inp:
      data = json.load(inp)
      pass

    ratioTest = data["test_ratio"]  # e.g. 0.2
    caseSet = CaseSet(ratio=ratioTest)

    trnSet_, tstSet = caseSet.splitSet()

    self.tstSet = tstSet

    # path of data used as training and possibly test
    matrix_data_path = data["train_data"]
    parent_dir = Path(__file__).parent.parent
    self.filePathH5 = parent_dir.joinpath(matrix_data_path)

    self.eval_file = parent_dir.joinpath(data["eval_file"])

    # this input is real
    # but should be normalized, before feed into the NN model
    inp = data["eval_inp"]
    eval_range = data["eval_range"]

    emin = eval_range["mins"][0]
    emax = eval_range["maxs"][0]
    inp[0] = (inp[0] - emin) / (emax-emin)

    emin = eval_range["mins"][1]
    emax = eval_range["maxs"][1]
    inp[1] = (inp[1] - emin) / (emax-emin)

    emin = eval_range["mins"][2]
    emax = eval_range["maxs"][2]
    inp[2] = (inp[2] - emin) / (emax-emin)

    self.eval_inp = inp
    pass  # end __init__

  def predict( self ):
  #-----------------------------------------------------------------------------
    # create a new empty h5 file to save the prediced data
    outH5Path = Path(self.eval_file)  # now called "./fnn.h5"
    h5 = h5py.File(outH5Path, 'w')
    h5.close()

    # predict and compare with the test set
    filePathH5 = self.filePathH5  # 'MatrixData.h5'
    tstSet = self.tstSet

    fields = ["T", "V", "P", "U", "W"]

    ifield = 0
    for var in fields:
      # check if the state dicts existed
      cur_dir = Path(__file__).parent
      stateDictsPath = cur_dir.joinpath("StateDicts")
      if not stateDictsPath.exists():
        stateDictsPath.mkdir(parents=True)

      var_dict_path = stateDictsPath.joinpath(f"dict_{var}.pth")

      if not var_dict_path.exists():
        var_dict_path = None
        print(f">>> Predicting Field {var}: TRIVAL!")
      else:
        print(f">>> Predicting Field {var}!")
        pass

      # create a Regression obj as model to do predicting
      R = Regression(var, var_dict_path)
      R.model.eval()  # only to predict

      # create a dataset obj
      fsDataset_test = FSimDataset(filePathH5, tstSet, var)
      #print(f"Test Cases are \n {tstSet}")

      # predict for the first case
      inp_, _, coords = fsDataset_test[0]

      # from user input, and already be normalized
      inp = torch.FloatTensor(self.eval_inp) # convert to torch.FloatTensor

      # the coordinates need to write only one time
      if ifield == 0:
        R.write2HDF(inp, outH5Path, coords=coords)
      else:
        R.write2HDF(inp, outH5Path, coords=None)
        pass

      ifield += 1
      pass  # end for self.predict
    pass  # end self.predict in class
  pass