"""
以下代码验证了采用 numpy.random 模块，实现独立随机数生成器的步骤

验证可行
"""
from numpy.random import Generator, PCG64

gen_unif = Generator(PCG64(seed=100))
gen_gaus = Generator(PCG64(seed=100))

a = gen_unif.random(1)
#b = gen_gaus.standard_normal(1)
c = gen_unif.random(1)

print(f"1st - 2nd: {a, c}")