
"""
This is main function to call:
  - train_p.py,
  - train_t.py,
  - train_u.py,
  - train_v.py, and
  - train_w.py
"""

from train_p import train_p

iSuccess = train_p(numOfEpochs=20)
print("Train Pres Successed? ", iSuccess)