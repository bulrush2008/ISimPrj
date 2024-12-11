
a = range(10)

for i in a:
  # do not print 5
  if i==5:
    continue

  print(i)

print("---- split ----")

for i in a:
  # 不打印 >=5 的元素
  if i==5:
    break

  print(i)
