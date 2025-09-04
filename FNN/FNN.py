
"""
This is main function to call

@author     @data       @aff        @version
Xia, S      2025.8.22   Simpop.cn   v6.x
"""

import argparse

from FNN_Train import FNN_train
from FNN_Eval  import FNN_Eval

if __name__=="__main__":
  """主函数
  """
  itrain   = False
  ipredict = False

  parser = argparse.ArgumentParser(description="训练 or 预测，随你挑")
  # 把 --train 和 --predict 都设成「布尔开关」
  parser.add_argument("--train",   action="store_true", help="启动训练")
  parser.add_argument("--predict", action="store_true", help="启动预测")

  args = parser.parse_args()

  if args.train:
    itrain = True

  if args.predict:
    ipredict = True

  if itrain:
    print("---------- Train ----------")
    fnn_train = FNN_train()

    chunk_iter = 3

    for var, epoch in fnn_train.train_info.items():
      N = epoch
      n = chunk_iter

      # 分割训练步骤
      steps = [n]*(N//n) + ([N%n] if N%n else [])
      for step in steps:
        messages = fnn_train.train_loop(var, step)
        print("> "+messages)

      print("") # 空行

  if ipredict:
    print("---------- Eval  ----------")
    fnn_eval = FNN_Eval()
    fnn_eval.predict()
    pass