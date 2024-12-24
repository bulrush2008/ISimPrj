
"""
This is main function to call:
  - split the data into train and test sets
  - train,
  - predict,
  - save to the database: .h5

@author     @data       @aff        @version
Xia, S      24.12.19    Simpop.cn   v3.x
"""

import h5py

from pathlib import Path

from Common.CaseSet import CaseSet
from Common.FSimDataset import FSimDataset
from train import train

#------------------------------------------------------------------------------
# case names list of train set and test set
# split the data, 49 = 39 + 10
ratioTest = 0.2
caseSet = CaseSet( ratio=ratioTest )

trnSet, tstSet = caseSet.splitSet()

# create a new empty h5 file to save the prediced data
h5 = h5py.File("./fnn.h5", 'w')
h5.close()

# path of data used as training and test
filePathH5 = Path("../FSCases/FSHDF/MatrixData.h5")
#aLive = filePathH5.exists()
#print(aLive)

#------------------------------------------------------------------------------
# train the fields one has assigned, which must belong in
# ["P", "T", "U", "V", "W"]

#varFields = ["T", "P", "U", "V", "W"]
#epochList = [3, 3, 2, 3, 2]
varFields = ["T", "P", "V"]

epochList = {"T":2, "P":2, "V":2}

print(f"*Fields {varFields} Will Be Model with Epochs {epochList}.")

models = train( epochList = epochList,
                fields    = varFields ,
                trainSet  = trnSet,
                testSet   = tstSet,
                dataPath  = filePathH5 )

#------------------------------------------------------------------------------
# plot loss history and save
dirPNG = Path("./Pics")

ifield = 0
for var in varFields:
  models[var].saveLossHistory2PNG(dirPNG)

  # predict and compare with the test set
  fsDataset_test = FSimDataset(filePathH5, tstSet, var)

  # for CXXX
  inp, _, coords = fsDataset_test[0]

  # the coordinates need to write only one time
  if ifield == 0:
    models[var].write2HDF(inp, Path("./fnn.h5"), coords=coords)
  else:
    models[var].write2HDF(inp, Path("./fnn.h5"), coords=None)

  ifield += 1
  pass
#------------------------------------------------------------------------------
# predict by the trained FNN model and write the data into database



