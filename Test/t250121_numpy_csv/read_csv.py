
import numpy as np

# 如果 'data.csv' 注释中，出现中文，那么比如制定编码方式，比如这里的 utf-8
data = np.loadtxt("./data.csv", delimiter=",", comments="#", encoding="utf-8")

print(data)