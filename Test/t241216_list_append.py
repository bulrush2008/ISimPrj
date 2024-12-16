# 一维列表
l = []
print(l, len(l))

l.append(1.0)
print(l, len(l))

# 二维列表初始化时，注意长度是 1，它已经包含了一个空列表
m = [[]]
print(m, len(m))

m.append([0,1])
print(m, len(m))

# 去除第一个元素：空列表
del(m[0])
print(m, len(m))