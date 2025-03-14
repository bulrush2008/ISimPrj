
"""
This is main function to call:
  - split the data into train and test sets
  - train,
  - predict, and save into the database: .h5

@author     @data       @aff        @version
Xia, S      2025.2.13   Simpop.cn   v6.x
"""
import sys

from FNN_Train import FNN_Train
from FNN_Eval import FNN_Eval

if __name__=="__main__":
#===============================================================================
  """
  - 主函数
  """
  args = sys.argv
  num_args = len(args)

  Train = False; Predict= False

  for arg in args:
    if arg == "--train"  : Train   = True
    if arg == "--predict": Predict = True
    pass

  if Train:
    print("---------- Train ----------")
    fnn_train = FNN_Train()
    fnn_train.train()
    pass

  if Predict:
    print("---------- Eval  ----------")
    fnn_eval = FNN_Eval()
    fnn_eval.predict()
    pass

  if not Train and not Predict:
    print("Trival, FNN Did Nothing.")
    pass
  pass # end "if __name__==..."