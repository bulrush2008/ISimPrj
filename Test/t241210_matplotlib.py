
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.arange(10)
y = np.sin(x)

ax.plot(x, y)

#fig.savefig("./t241210_matplotlib.png")