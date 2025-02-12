
import sys

# 获取命令行参数
args = sys.argv

# 获取参数的数目
num_args = len(args)

# 打印参数数目
print(f"Number of arguments: {num_args}")

# 打印所有参数
print("Arguments:")
for i, arg in enumerate(args):
    print(f"Argument {i}: {arg}")