
"""
This is main function to call

@author     @data       @aff        @version
Xia, S      2025.8.22   Simpop.cn   v6.x
"""

import argparse

from FNN_train import FNN_train
from FNN_eval  import FNN_eval

if __name__=="__main__":
  """
  解析命令参数，根据具体命令参数，选择训练或预测模式
  """

  # 创建解析器对象
  parser = argparse.ArgumentParser(description="训练 or 预测，随你挑")

  # 把 --train 和 --predict 都设成「布尔开关」
  parser.add_argument("--train",   action="store_true", help="启动训练")
  parser.add_argument("--predict", action="store_true", help="启动预测")

  # 解析命令行参数
  args = parser.parse_args()

  if args.train:
    print("---------- Train ----------")
    fnn_train = FNN_train()

    chunk_iter = 1
    print(f"> chunk_iter: {chunk_iter}")

    for var, epoch in fnn_train.train_info.items():
      N = epoch
      n = chunk_iter

      # 分割训练步骤
      steps = [n]*(N//n) + ([N%n] if N%n else [])
      for step in steps:
        messages = fnn_train.train_loop(var, step)
        print("> "+messages)

      print("") # 空行

  if args.predict:
    print("---------- Eval  ----------")
    fnn_eval = FNN_eval()
    fnn_eval.predict()
    pass