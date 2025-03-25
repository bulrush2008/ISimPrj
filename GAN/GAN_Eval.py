
"""
This is function to call:
  - split the data into train and test sets
  - train,
  - predict,
  - save to the database: .h5

@author     @data       @aff        @version
Xia, S      2025.2.21   Simpop.cn   v3.x
"""
import sys
import json

import h5py
import torch

from pathlib import Path

from Common.CaseSet import CaseSet
from Common.FSimDataset import FSimDataset
from Common.Generator import Generator
from Common.Discriminator import Discriminator

class GAN_Eval(object):
#===============================================================================
  def __init__( self ):
  #-----------------------------------------------------------------------------
    # split the cases into train and test sets
    # now: 125 = 100 + 25
    with open("./GAN_Eval.json", 'r') as inp:
      data = json.load(inp)
      pass

    ratioTest = data["test_ratio"]  # e.g. 0.2
    caseSet = CaseSet(ratio=ratioTest)

    trnSet_, tstSet = caseSet.splitSet()
    self.tstSet = tstSet

    # path of data used as training and possibly test
    matrix_data_path = data["train_data"]
    self.filePathH5 = Path(matrix_data_path)

    self.eval_file = data["eval_file"]

    # this input is real
    # but should be normalized, before feed into the NN model
    inp = data["eval_inp"]

    inp[0] = (inp[0] - 400.0) / 100.0
    inp[1] = inp[1] - 1.5
    inp[2] = inp[2] / 1000000.0

    self.eval_inp = inp
    pass  # end __init__

  def predict( self ):
  #-----------------------------------------------------------------------------
    # create a new empty h5 file to save the prediced data
    outH5Path = Path(self.eval_file)

    h5 = h5py.File(outH5Path, 'w')
    h5.close()

    # predict and compare with the test set
    filePathH5 = self.filePathH5  # 'MatrixData.h5'
    tstSet = self.tstSet

    fields = ["T", "V", "P", "U", "W"]

    ifield = 0
    for var in fields:
      # create a Generator object as model, from the state_dict
      # gen a obj as regression, and then train the model
      stateDictsPath = Path("StateDicts")
      var_dict_path = stateDictsPath.joinpath(f"G_Dict_{var}.pth")

      if not var_dict_path.exists():
        var_dict_path = None
        print(f">>> Predicting Field {var}: TRIVAL!")
      else:
        print(f">>> Predicting Field {var}!")
        pass

      G = Generator(var, var_dict_path)
      G.model.eval()  # predict model

      fsDataset_test = FSimDataset(filePathH5, tstSet, var)

      # predict for the first case
      inp_, _, coords = fsDataset_test[0]

      # from user input, and already be normalized
      inp = torch.FloatTensor(self.eval_inp) # convert to torch.FloatTensor

      # the coordinates need to write only one time
      if ifield == 0:
        G.write2HDF(inp, outH5Path, coords=coords)
      else:
        G.write2HDF(inp, outH5Path, coords=None)
        pass

      ifield += 1
      pass  # end for-loop
    pass  # end function
  pass  # end class


if __name__=="__main__":
#===============================================================================
  print("Only define class 'GAN_Eval' here")
  pass