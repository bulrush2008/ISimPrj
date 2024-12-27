

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
    pass

  def predict(self):
  #----------------------------------------------------------------------------
    pass  # member func predict end
  pass  # class GAN end

if __name__=="__main__":
  # create a model to train and predict
  gan = GAN()

  # Start training the models
  print("---------- Train ----------")
  gan.train()

  print("---------- Eval  ----------")
  gan.predict()


