

import torch.nn as nn
import pandas

# discriminator class

class Discriminator(nn.Module):

  def __init__(self):
    # initialise parent pytorch class
    super().__init__()    
    # define neural network layers
    self.model = nn.Sequential(
      nn.Linear(125557, 1000),
      nn.LeakyReLU(),
      nn.LayerNorm(100),

      nn.Linear(1000, 300),
      nn.LeakyReLU(),
      nn.LayerNorm(300),

      nn.Linear(300,100),
      nn.LeakyReLU(),
      nn.LayerNorm(100),

      nn.Linear(1)
    )
        
    # create loss function
    self.loss_function = nn.BCELoss()

    # create optimiser, simple stochastic gradient descent
    self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    # counter and accumulator for progress
    self.counter = 0
    self.progress = []
    pass

  def forward(self, inputs):
    # simply run model
    return self.model(inputs)

  def train(self, inputs, targets):
    # calculate the output of the network
    outputs = self.forward(inputs)

    # calculate loss
    loss = self.loss_function(outputs, targets)

    # increase counter and accumulate error every 10
    self.counter += 1;
    if (self.counter % 10 == 0):
      self.progress.append(loss.item())
      pass
    if (self.counter % 1000 == 0):
      print("counter = ", self.counter)
      pass

    # zero gradients, perform a backward pass, update weights
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  def plot_progress(self):
    df = pandas.DataFrame(self.progress, columns=['loss'])
    df.plot()
    pass
  pass