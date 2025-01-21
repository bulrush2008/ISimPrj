
"""
This is main function to call:
  - split the data into train and test sets
  - train,
  - predict,
  - save to the database: .h5

@author     @data       @aff        @version
Xia, S      2025.1.17   Simpop.cn   v5.x
"""
import sys
import json

import h5py
import torch

from pathlib import Path

from Common.CaseSet import CaseSet
from Common.FSimDataset import FSimDataset
from Common.Regression  import Regression

class FNN(object):
  #----------------------------------------------------------------------------
  def __init__( self ):
    # split the cases into train and test sets
    # now: 125 = 100 + 25
    with open("./FNN.json", 'r') as inp:
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

    self.eval_file = data["eval_file"]
    pass

  def train( self ):
    #--------------------------------------------------------------------------
    # train the fields one has assigned, which must be in
    # ["P", "T", "U", "V", "W"]
    # the order in list does not matter
    #fieldList = {"T":1, "V":1}
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

  def predict( self ):
    # create a new empty h5 file to save the prediced data
    outH5Path = Path(self.eval_file)  # now called "./fnn.h5"
    h5 = h5py.File(outH5Path, 'w')
    h5.close()

    # predict and compare with the test set
    filePathH5 = self.filePathH5  # 'MatrixData.h5'
    tstSet = self.tstSet

    fields = ["T", "V", "P", "U", "W"]

    """
    2025.1.20，新的修改思路
    先实现，再修改！
    """

    ifield = 0
    for var in fields:
      # create a Regression obj as model, from the state_dict
      # gen a obj as regression, and then train the model
      stateDictsPath = Path("StateDicts")
      var_dict_path = stateDictsPath.joinpath(f"dict_{var}.pth")

      if not var_dict_path.exists():
        var_dict_path = None
        print(f">>> Predicting Field {var}: TRIVAL!")
        #print(f"  - State Dictionary 'dict_{var}.pth' Not Exist")
      else:
        print(f">>> Predicting Field {var}!")
        pass

      R = Regression(var, var_dict_path)
      R.model.eval()  # only to predict

      # create a dataset obj
      fsDataset_test = FSimDataset(filePathH5, tstSet, var)
      #print(f"Test Cases are \n {tstSet}")

      # predict for the first case
      inp, _, coords = fsDataset_test[24]

      # the coordinates need to write only one time
      if ifield == 0:
        R.write2HDF(inp, outH5Path, coords=coords)
      else:
        R.write2HDF(inp, outH5Path, coords=None)
        pass

      ifield += 1
      pass
    pass

  def _train( self,
              varList :dict,
              trainSet:list,
              testSet :list,
              dataPath:Path )->dict:
    """
    Train the FNN model by a give trainset, in which some cases field included.
    - varList : dict of epochs for each field, such as ["P":1,"T":2]
    - trainSet: list of case names in train set, each is a string
    - testSet : list of case names in test set, each is a string
    - dataPath: path of data of train set
    """

    #--------------------------------------------------------------------------
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
      fsDataset_train = FSimDataset(dataPath, trainSet, var)

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
        print(f"  - Training Epoch {i+1} of {epochs} for {var}")
        for inp, label, _ in fsDataset_train:
          R.train(inp, label)
          pass
        pass

      models[var] = R
      pass

    # now all variable models have been trained
    return models
  pass

if __name__=="__main__":
  # create a model to train and predict
  fnn = FNN()

  # Start training the models
  #print("---------- Train ----------")
  #fnn.train()

  print("---------- Eval  ----------")
  fnn.predict()