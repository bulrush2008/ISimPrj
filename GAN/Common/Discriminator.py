
import torch
import torch.nn as nn
import pandas
import math

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

    # initialize weights，using He Kaiming method now
    self._initialize_weights()

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
    self.progress.append(loss.item()) # item() 用于取元素

    if self.counter%10 == 0:
      print(f"    D- {self.counter} Cases Trained for {self.varName} ...")
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

  def _initialize_weights(self):
    """
    - inner function, call only once at initialization
    - configure initial model weights
    """
    for m in self.model:
      if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
          nn.init.zeros_(m.bias)
          pass
        pass
      pass  # for-loop end
    pass  # func '_initialize_weights' end
  pass  # Class End

if __name__=="__main__":
  d = Discriminator(varName="T")
  pass