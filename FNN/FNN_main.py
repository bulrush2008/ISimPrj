
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

from train import train

#------------------------------------------------------------------------------
# case names list of train set and test set
# split the data, 49 = 40 + 9
ratioTest = 0.2

sizeOfTestSet = np.int64( numOfAllCases*ratioTest )

# 42 is random seed，other ints also work
np.random.seed(42)
permut = np.random.permutation( numOfAllCases )

# names list: case list consist of train set
listTestCase = []
for i in permut[:sizeOfTestSet]:
  theCase = "C" + "%03d"%idxList[i]
  listTestCase.append(theCase)

# names list: case list consist of test set
listTrainCase = []
for i in permut[sizeOfTestSet:]:
  theCase = "C" + "%03d"%idxList[i]
  listTrainCase.append(theCase)

# create a new empty h5 file
h5 = h5py.File("./fnn.h5", 'w')
h5.close()

#------------------------------------------------------------------------------
# train the fields one has assigned, which must belong in ["P", "T", "U", "V", "W"]

iSuccess = train( numOfEpochs=1 )
print("Train Pres Successed? ", iSuccess)

#------------------------------------------------------------------------------
# predict by the trained FNN model


#------------------------------------------------------------------------------
# write the predicted fields onto database, i.e. a new hdf5 file

