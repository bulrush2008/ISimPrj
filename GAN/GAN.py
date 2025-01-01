
"""
This is main function to call:
  - split the data into train and test sets
  - train,
  - predict,
  - save to the database: .h5

@author     @data       @aff        @version
Xia, S      2024.12.31  Simpop.cn   v2.x
"""
import sys

import h5py
import torch

from pathlib import Path

from Common.CaseSet import CaseSet
from Common.FSimDataset import FSimDataset
from Common.Generation  import Generation

class GAN(object):
  #----------------------------------------------------------------------------
  def __init__( self ):
    # split the cases into train and test sets
    # now: 49 = 39 + 10
    ratioTest = 0.2
    caseSet = CaseSet( ratio=ratioTest )

    trnSet, tstSet = caseSet.splitSet()

    self.trnSet = trnSet
    self.tstSet = tstSet

    # path of data used as training and possibly test
    self.filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")
    pass

  def train( self ):
    #--------------------------------------------------------------------------
    # train the fields one has assigned, which must be in
    # ["P", "T", "U", "V", "W"]
    # the order in list does not matter
    fieldList = {"T":2}

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
    outH5Path = Path("./fnn.h5")
    h5 = h5py.File(outH5Path, 'w')
    h5.close()

    # predict and compare with the test set
    filePathH5 = self.filePathH5
    tstSet = self.tstSet

    fields = ["T", "V", "P", "U", "W"]

    ifield = 0
    for var in fields:
      # create a Generation object as model, from the state_dict
      # gen a obj as regression, and then train the model
      stateDictsPath = Path("StateDicts")
      var_dict_path = stateDictsPath.joinpath(f"dict_{var}.pth")

      if not var_dict_path.exists():
        var_dict_path = None
        print(f"! Eval Warn: Predict {var} is TRIVAL!")
        print(f"  >>> State Dictionary 'dict_{var}.pth' Not Exist")
      else:
        print(f">>> Hi, Now We are Predicting Field {var}!")
        pass

      G = Generation(var, var_dict_path)
      G.model.eval()  # predict model

      fsDataset_test = FSimDataset(filePathH5, tstSet, var)

      # predict for the first case
      inp, _, coords = fsDataset_test[0]

      # the coordinates need to write only one time
      if ifield == 0:
        G.write2HDF(inp, outH5Path, coords=coords)
      else:
        G.write2HDF(inp, outH5Path, coords=None)
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

      G = Generation(var, var_dict_path)

      print(f"*Now we are training {var} field:")

      # train the model
      epochs = varList[var]

      for i in range(epochs):
        print(f"  - Training Epoch {i+1} of {epochs} for {var}")
        for inp, label, _ in fsDataset_train:
          G.train(inp, label)
          pass
        pass

      models[var] = G
      pass

    # now all variable models have been trained
    return models
  pass

if __name__=="__main__":
  # create a model to train and predict
  gan = GAN()

  # Start training the models
  print("---------- Train ----------")
  gan.train()

  print("---------- Eval  ----------")
  gan.predict()