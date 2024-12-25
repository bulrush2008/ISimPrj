
"""
This is main function to call:
  - split the data into train and test sets
  - train,
  - predict,
  - save to the database: .h5

@author     @data       @aff        @version
Xia, S      24.12.19    Simpop.cn   v4.x
"""

import h5py
import torch

from pathlib import Path

from Common.CaseSet import CaseSet
from Common.FSimDataset import FSimDataset
from Common.Regression  import Regression

class FNN(object):
  #----------------------------------------------------------------------------
  def __init__(self):
    # split the cases into train and test sets
    # now: 49 = 39 + 10
    ratioTest = 0.2
    caseSet = CaseSet( ratio=ratioTest )

    trnSet, tstSet = caseSet.splitSet()

    self.trnSet = trnSet
    self.tstSet = tstSet
    pass

  def train(self):

    # path of data used as training and test
    filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")
    #aLive = filePathH5.exists()
    #print(aLive)

    #--------------------------------------------------------------------------
    # train the fields one has assigned, which must belong in
    # ["P", "T", "U", "V", "W"]

    fieldList = {"T":1}

    print(f"*Fields Models Will Be Trained with Epochs {fieldList}.")

    trnSet = self.trnSet
    tstSet = self.tstSet

    models = self._train( epochList = fieldList,
                          trainSet  = trnSet,
                          testSet   = tstSet,
                          dataPath  = filePathH5 )

    #--------------------------------------------------------------------------
    dirPNG = Path("./Pics")

    #ifield = 0
    for var in fieldList.keys():
      #------------------------------------------------------------------------
      # plot loss history and save
      models[var].saveLossHistory2PNG(dirPNG)

      #------------------------------------------------------------------------
      # predict and compare with the test set
      #fsDataset_test = FSimDataset(filePathH5, tstSet, var)

      # for CXXX
      #inp, _, coords = fsDataset_test[0]

      #------------------------------------------------------------------------
      # the coordinates need to write only one time
      #if ifield == 0:
      #  models[var].write2HDF(inp, outH5Path, coords=coords)
      #else:
      #  models[var].write2HDF(inp, outH5Path, coords=None)

      #ifield += 1

      #------------------------------------------------------------------------
      # save model parameters
      model_dicts_name = Path(f"./ModelDict/dict_{var}.pth")
      torch.save(models[var].model.state_dict(), model_dicts_name)
    pass

  def predict(self):
    # create a new empty h5 file to save the prediced data
    outH5Path = Path("./fnn.h5")
    h5 = h5py.File(outH5Path, 'w')
    h5.close()
    pass

  def _train( self,
              epochList:dict,
              trainSet :list,
              testSet  :list,
              dataPath :Path )->dict:
    """
    Train the FNN model by a give trainset, in which some cases field included.
    - epochList: dict of epochs for each field, such as ["P":1,"T":2]
    - fields   : list of variable names, such as ["P", "U"]
    - trainSet : list of case names in train set, each is a string
    - testSet  : list of case names in test set, each is a string
    """

    #--------------------------------------------------------------------------
    # extract the var names
    fields = []
    for key in epochList.keys():
      fields.append(key)
      pass
  
    # including all trained models
    models = {}

    # train fields
    for var in fields:
      fsDataset_train = FSimDataset(dataPath, trainSet, var)

      # gen a obj as regression, and then train the model

      var_dict_path = Path(f"./ModelDict/dict_{var}.pth")

      if not var_dict_path.exists():
        var_dict_path = None
        print(f"!Warning: File 'dict_{var}.pth' Not Exist.")
        pass

      R = Regression(var, var_dict_path)

      print(f"*Now we are training {var} field:")

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

    # now all variable models have been trained
    return models
  pass

if __name__=="__main__":
  fnn = FNN()

  fnn.train()