
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)

base = np.arange(0,100)
#print(base)

x = np.random.random(100)*10.0 + base
y = np.random.random(100)*10.0 + base

m = max(max(x), max(y))
im = range(int(m))

fig, ax = plt.subplots(1,1)
ax.plot(x,y, ls='', marker='o', markersize=2, markeredgecolor='black', markerfacecolor='black', label="Regression")
ax.plot(im,im)

ax.legend()

fig.savefig("./reg.png")