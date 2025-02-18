
import matplotlib.pyplot as plt
import numpy as np


class PlotData(object):
  def __init__(self,  x:np.ndarray,
                      y1:np.ndarray,
                      y2:np.array,
                      info:list)->None:
    self.x = x
    self.y1 = y1
    self.y2 = y2

    # info = [fig_name, fig1_title, fig2_title, fig3_title]
    self.info = info
    pass
  def plot(self):
    x = self.x
    y1 = self.y1
    y2 = self.y2

    info = self.info

    figName = info[0]
    title01 = info[1]
    title02 = info[2]
    title03 = info[3]

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(21,6))

    ax1.plot(x,y1, ls='-', marker='o', markerfacecolor='w', markersize=8, label="CFD")
    ax1.plot(x,y2, ls='-', marker='d', markerfacecolor='w', markersize=8, label="FSim")
    ax1.set_ylabel("Temperature (K)")
    ax1.set_xlabel("Case Number")
    ax1.legend()
    ax1.set_title(title01)

    abs_err = abs(y1-y2)
    ax2.plot(x, abs_err, marker='o', markersize=8, label="Abs Error (K)")
    ax2.set_ylabel("Temperature Difference (K)")
    ax2.set_xlabel("Case number")
    ax2.legend()
    ax2.set_title(title02)

    rel_err = abs(y1-y2) / y1 * 100.
    ax3.plot(x, rel_err, marker='o', markersize=8, label="Rel Error")
    ax3.set_ylabel("Relative Error (%)")
    ax3.set_xlabel("Case number")
    ax3.legend()
    ax3.set_title(title03)

    plt.savefig(figName, dpi=100)
    pass
  pass

if __name__=="__main__":
  data = np.loadtxt("eval_T_max.csv", delimiter=',', dtype=float, encoding="utf-8")

  x = data[:,0]

  y1 = data[:,1]
  y2 = data[:,2]
  y3 = data[:,3]
  y4 = data[:,4]

  # plot the data about "maximum temperature over field"
  p1_info = [ "./Eval-T-max-1.png",
              "Max T over Field",
              "Temp Diff over Field",
              "Temp Diff over Field"]
  p1 = PlotData(x, y1, y3, p1_info)
  p1.plot()

  # plot the data about "maximum temperature over z-mid plane"
  p2_info = [ "./Eval-T-max-2.png",
              "Max T over z-mid Plane",
              "Temp Diff over z-mid Plane",
              "Temp Diff over z-mid Plane"]
  p2 = PlotData(x, y2, y4, p2_info)
  p2.plot()



