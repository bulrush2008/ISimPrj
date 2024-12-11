
import pandas
import matplotlib.pyplot as plt
import numpy as np

#data = np.array([1.0, 5.1, 4.3, 1.5, 6.2])
data = [1.0, 5.1, 4.3, 1.5, 6.2]

df = pandas.DataFrame(data, columns=["test"])

ax = df.plot()

ax.figure.savefig("t241211_dataframe_plot.png")
#plt.show()

