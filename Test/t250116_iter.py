import sys

# 一个简单列表数据，一个 iterable
lst = [2,5,7,10]

# 这样打印列表
for i in lst:
  print(i)

# list obj is not an iterator
#while True:
#  print(next(lst))

# 将列表变为可迭代对象：有必要？iterator 不同于 iterable，记录了迭代状态
il = iter(lst)

# 打印此可迭代对象
while True:
  try:
    print(next(il))
  except StopIteration:
    print("end")
    sys.exit()