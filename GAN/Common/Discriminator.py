
import torch.nn as nn

class Discriminator(object):
  def __init__(self):
    # init by parent init method
    super().__init__()

    # define neural network layers
    self.model = nn.Sequential(
      nn.Linear(125557,1000),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(),

      nn.Linear(1000,300),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(),

      nn.Linear(300,100),
      nn.LeakyReLU(0.02),
      nn.LayerNorm(),

      nn.Linear(100,1),
    )

    # create a loss function
    self.loss_function = nn.BCELoss()

    # create optimizer, Adam method adopted
    self.optimiser = torch.optim.Adam(self.parameter(), lr=0.0001)

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

  def plot_progress(self):
    pass
  pass