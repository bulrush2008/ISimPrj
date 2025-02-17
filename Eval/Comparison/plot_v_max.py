
import matplotlib.pyplot as plt
import numpy as np

#file = open("./eval_v_max.csv", 'r')

data = np.loadtxt("eval_v_max.csv", delimiter=',', dtype=float, encoding="utf-8")

#print(len(data))

fig, axes = plt.subplots(1,2, figsize=(15,6))

ax = axes[0]

x = data[:,0]
y = data[:,1]
z = data[:,2]
ax.plot(x,y, ls='-', marker='o', markerfacecolor='w', markersize=8, label="CFD")
ax.plot(x,z, ls='-', marker='d', markerfacecolor='w', markersize=8, label="FSim")

ax.set_ylabel("Max v")
ax.set_xlabel("Case Number")

ax.legend()

ax = axes[1]

a = (data[:,2] - data[:,1]) / data[:,1] * 100.0
a = abs(a)
ax.plot(x,a, marker="o", color="black", markersize=10, markerfacecolor="w", label="Rel Err (%)")
ax.set_ylabel("Rel Error (%)")
ax.legend(loc=4)

#b = data[:,2] - data[:,1]
#ax = ax.twinx()
#ax.plot(x,b, marker="^", markerfacecolor='w', markersize=8, label="Err (K)")

ax.set_xlabel("Case Number")
ax.set_ylabel("Error")

ax.legend()

plt.savefig("./Eval-Err-v.png", dpi=100)