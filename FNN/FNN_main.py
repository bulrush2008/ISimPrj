
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

from train_p import train_p
from train_t import train_t
from train_u import train_u
from train_v import train_v
from train_w import train_w

# create a new empty h5 file
h5 = h5py.File("./fnn.h5", 'w')
h5.close()

iSuccess = train_p(numOfEpochs=1)
print("Train Pres Successed? ", iSuccess)

iSuccess = train_t(numOfEpochs=1)
print("Train Temp Successed? ", iSuccess)

iSuccess = train_u(numOfEpochs=1)
print("Train UVel Successed? ", iSuccess)

iSuccess = train_v(numOfEpochs=1)
print("Train VVel Successed? ", iSuccess)

iSuccess = train_w(numOfEpochs=1)
print("Train WVel Successed? ", iSuccess)