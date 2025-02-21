
"""
This is main function to call:
  - split the data into train and test sets
  - train,
  - predict,
  - save to the database: .h5

@author     @data       @aff        @version
Xia, S      2025.2.21   Simpop.cn   v3.x
"""
import sys

from GAN_Train import GAN_Train
from GAN_Eval import GAN_Eval

if __name__=="__main__":
  args = sys.argv
  num_args = len(args)

  Train = False; Predict= False

  for arg in args:
    if arg == "--train"  : Train   = True
    if arg == "--predict": Predict = True
    pass

  if Train:
    print("---------- Train ----------")
    gan_train = GAN_Train()
    gan_train.train()
    pass

  if Predict:
    print("---------- Eval  ----------")
    gan_eval = GAN_Eval()
    gan_eval.predict()
    pass

  if not Train and not Predict:
    print("Trival, GAN Did Nothing.")
    pass
  pass  # end 'if __name__=="__main__":'