
"""
测试 numpy 操作的效率

来自：每训练周期后的误差统计
"""

import numpy as np
import time

size = 125000

np.random.seed(42)

r1 = np.random.random(size)
r2 = np.random.random(size)

#print(r1[10:20])
#print(r2[10:20])

sta = time.perf_counter()

for j in range(100):
  e = 0.0
  #for i in range(size):
  #  e += abs(r1[i] - r2[i])
  #  pass

  r3 = abs(r1 - r2)
  e = sum(r3)
  print(f"e = {e}")
  pass

end = time.perf_counter()

print(f"run time = {end-sta}")

