
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
# train the fields one has assigned, which must belong in
# ["P", "T", "U", "V", "W"]

varFields = ["T", "P", "U", "V", "W"]
epochList = [3, 3, 2, 3, 2]

iSuccess = train( epochList = epochList,
                  fields    = varFields ,
                  trainSet  = trnSet,
                  testSet   = tstSet )

#print("Train Pres Successed? ", iSuccess)

#------------------------------------------------------------------------------
# predict by the trained FNN model and write the data into database



