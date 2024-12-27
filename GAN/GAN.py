
"""
This is main function to call:
  - split the data into train and test sets
  - train,
  - predict,
  - save to the database: .h5

@author     @data       @aff        @version
Xia, S      24.12.27    Simpop.cn   v2.x
"""

from pathlib import Path

from Common.CaseSet import CaseSet

class GAN(object):
#==============================================================================
  def __init__(self):
  #----------------------------------------------------------------------------
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

  def train(self):
  #----------------------------------------------------------------------------
    # train the fields one has assigned, which must be in
    # ["P", "T", "U", "V", "W"], and
    # the order in list does not matter
    fieldList = {"T":5}

    print(f"*Fields Models Will Be Trained with Epochs {fieldList}.")

    trnSet = self.trnSet
    tstSet = self.tstSet

    filePathH5 = self.filePathH5

    models = self._train( varList  = fieldList,
                          trainSet = trnSet,
                          testSet  = tstSet,
                          dataPath = filePathH5 )

    dirPNG = Path("./Pics")

    for var in fieldList.keys():
      #------------------------------------------------------------------------
      # plot loss history and save
      models[var].saveLossHistory2PNG(dirPNG)

      #------------------------------------------------------------------------
      # save model parameters
      model_dicts_name = Path(f"./ModelDict/dict_{var}.pth")
      torch.save(models[var].model.state_dict(), model_dicts_name)
      pass
    pass

  def predict(self):
  #----------------------------------------------------------------------------
    pass  # member func predict end

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

    # extract the var names
    fields = []
    for key in varList.keys():
      fields.append(key)
      pass

    # including all trained models
    models = {}

    pass  # private member func '_train()' ends
  pass  # class GAN end

if __name__=="__main__":
  # create a model to train and predict
  gan = GAN()

  # Start training the models
  print("---------- Train ----------")
  gan.train()

  print("---------- Eval  ----------")
  gan.predict()


