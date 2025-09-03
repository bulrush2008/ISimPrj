
"""
This is main function to call

@author     @data       @aff        @version
Xia, S      2025.8.22   Simpop.cn   v6.x
"""
import sys

from FNN_Train import FNN_Train
from FNN_Eval  import FNN_Eval

if __name__=="__main__":
#===============================================================================
  """主函数
  """
  args = sys.argv
  num_args = len(args)

  Train   = False
  Predict = False

  for arg in args:
    if arg == "--train":
      Train   = True
    if arg == "--predict":
      Predict = True

  if Train:
    print("---------- Train ----------")
    fnn_train = FNN_Train()

    for var, epoch in fnn_train.train_info.items():
      chunk_iter = 2
      while True:
        istep, epoch = fnn_train.train_loop(var, chunk_iter)

        print(f"> Current step: {istep}/{epoch}")

        if istep >= epoch:
          print(f"{var} training over.\n")
          break

  if Predict:
    print("---------- Eval  ----------")
    fnn_eval = FNN_Eval()
    fnn_eval.predict()
    pass

  if not Train and not Predict:
    print("... FNN Have Done Nothing ...")