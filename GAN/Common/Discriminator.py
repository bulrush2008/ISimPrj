
import torch
import torch.nn as nn
import pandas

from pathlib import Path

class Discriminator(nn.Module):
  def __init__(self, varName:str):
    # init by parent init method
    super().__init__()

    if varName not in ["P", "U", "V", "W", "T"]:
      raise ValueError("Error in D: Var Name in [P,U,V,W,T]")

    self.varName = varName

    # define neural network layers
    self.model = nn.Sequential(
      nn.Linear(125557,1000),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(1000),

      nn.Linear(1000,300),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(300),

      nn.Linear(300,100),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(100),

      nn.Linear(100,1),
    )

    # create a loss function
    self.loss_function = nn.BCELoss()

    # create optimizer, Adam method adopted
    self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    # counter and accumulator for progress
    self.counter = 0
    self.progress = []
    pass

  def forward(self, inputs):
    # simply run model
    return self.model(inputs)
    pass

  def train(self, inputs, targets):
    # calculate the output of the network
    outputs = self.forward(inputs)

    # calculate loss
    loss = self.loss_function(outputs, targets)

    # increase counter and accumulate error every 10
    self.counter += 1

    if self.counter%10 == 0:
      self.progress.append(loss.item()) # item() 用于取元素
      pass

    if self.counter%1000 == 0:
      print("Counter = ", self.counter)
      pass

    # zero gradient, perform a backward pass, update weights
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  def saveLossHistory2PNG(self, outDir:Path)->None:
    df = pandas.DataFrame(self.progress, columns=["Loss"])
    ax = df.plot( title  = f"D-Loss history of {self.varName}",
                  color  = "black",
                  xlabel = "Number of Trained Cases",
                  ylabel = "Loss Value",
                  logy   = True)
    outFile = outDir.joinpath(f"D_lossHistory_{self.varName}.png")
    ax.figure.savefig(outFile)
    pass
  pass

if __name__=="__main__":
  d = Discriminator(varName="T")
  pass