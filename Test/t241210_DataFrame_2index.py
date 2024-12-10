
import pandas as pd

a = [[1,2],[4,5],[11,12]]

df = pd.DataFrame(a, index=["r0","r1","r2"], columns=["c0","c1"])
#print(df)

#x = df.iloc[0,1]
#print(x)

#y = df.loc['r0','c0']
#print(y)

z = df["c0"]; print(z, type(z))
print(z["r0"],z.iloc[0])
print(type(z["r0"]))

#print(df["c0", "r0"])  # wrong