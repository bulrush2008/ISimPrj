
import pandas as pd

l = [2, 5.5, 4.3, 4.1, -2.0, -1.0]

ds = pd.Series(l)
print("Series:\n", ds)

# data format conversion: a list to DataFrame
df = pd.DataFrame(l, columns=["L"]) # note the square brackets "[]"

# DataFrame can directly plot its data
ax = df.plot()

ax.figure.savefig("t241210_list_dataframe.png")

print("DataFrame:\n", df)