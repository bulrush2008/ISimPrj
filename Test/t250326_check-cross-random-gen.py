
"""
考察交叉使用 numpy 不同的随机数生成器是否相互影响
比如较低调用 numpy.random.random & numpy.random.randn()
"""

import numpy as np

np.random.seed(42)

a = np.random.random(1)
#b = np.random.randn(1)
c = np.random.random(1)

print(f"1st & 2nd = {a,c}")