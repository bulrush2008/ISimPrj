
"""
This is main function to call:
  - train_p.py,
  - train_t.py,
  - train_u.py,
  - train_v.py, and
  - train_w.py
"""

from train_p import train_p
from train_t import train_t
from train_u import train_u
from train_v import train_v
from train_w import train_w

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