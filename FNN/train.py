
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
def train(epochList:dict, fields:list, trainSet:list, testSet:list ):
  """
  Train the FNN model by a give trainset, in which some cases field included.
  - epochList: dict of epochs for each field, such as ["P":1,"T":2]
  - fields   : list of variable names, such as ["P", "U"]
  - trainSet : list of case names in train set, each is a string
  - testSet  : list of case names in test set, each is a string
  """

  #----------------------------------------------------------------------------
  # init of class of database
  filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")
  #aLive = filePathH5.exists()
  #print(aLive)

  #----------------------------------------------------------------------------
  # train fields

  # including all trained models
  models = {}

  for var in fields:
    fsDataset_train = FSimDataset(filePathH5, trainSet, var)

    # gen a obj as regression, and then train the model
    R = Regression(var)

    print(f"*Now we train the {var} field:")

    # train the model
    epochs = epochList[var]

    for i in range(epochs):
      print(f"  - Training Epoch {i+1} of {epochs} for {var}")
      for inp, label, _ in fsDataset_train:
        R.train(inp, label)
        pass
      pass

    models[var] = R
    pass

  return models