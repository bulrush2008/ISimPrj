
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

from Common.CaseSet import CaseSet

from train import train

#------------------------------------------------------------------------------
# case names list of train set and test set
# split the data, 49 = 39 + 10
ratioTest = 0.2
caseSet = CaseSet( ratio=ratioTest )

trnSet, tstSet = caseSet.splitSet()

# create a new empty h5 file
h5 = h5py.File("./fnn.h5", 'w')
h5.close()

#------------------------------------------------------------------------------
# train the fields one has assigned, which must belong in ["P", "T", "U", "V", "W"]

varfiels = ["T", "P"]
iSuccess = train( numOfEpochs=1, fields=varfiels , trainSet=trnSet, testSet=tstSet )
#print("Train Pres Successed? ", iSuccess)

#------------------------------------------------------------------------------
# predict by the trained FNN model


#------------------------------------------------------------------------------
# write the predicted fields onto database, i.e. a new hdf5 file

