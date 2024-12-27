

# import libraries

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import h5py
from pathlib import Path
import pandas, numpy, random
import matplotlib.pyplot as plt

# dataset class

class CelebADataset(Dataset):
#==============================================================================
  def __init__(self, file):
    self.file_object = h5py.File(file, 'r')
    self.dataset = self.file_object['img_align_celeba']
    pass
    
  def __len__(self):
    return len(self.dataset)
    
  def __getitem__(self, index):
    if (index >= len(self.dataset)):
      raise IndexError()
    img = numpy.array(self.dataset[str(index)+'.jpg'])
    return torch.FloatTensor(img) / 255.0
    
  def plot_image(self, index):
    plt.imshow(numpy.array(self.dataset[str(index)+'.jpg']), interpolation='nearest')
    plt.show()
    pass
  pass

# create Dataset object

celebaDir = Path('../FSCases/CelebA/celeba_aligned_small.h5py')
#print(f"the celebA Data exists? {celebaDir.exists()}")
celeba_dataset = CelebADataset(celebaDir)

# check data contains images

#celeba_dataset.plot_image(43)

# functions to generate random data

def generate_random_image(size):
#==============================================================================
  random_data = torch.rand(size)
  return random_data

def generate_random_seed(size):
#==============================================================================
  random_data = torch.randn(size)
  return random_data

class View(nn.Module):
#==============================================================================
  def __init__(self, shape):
    super().__init__()
    self.shape = shape,
    pass

  def forward(self, x):
    return x.view(*self.shape)

# discriminator class
class Discriminator(nn.Module):
#==============================================================================
  def __init__(self):
  #----------------------------------------------------------------------------
    # initialise parent pytorch class
    super().__init__()
        
    # define neural network layers
    self.model = nn.Sequential(
      View(218*178*3),
      nn.Linear(3*218*178, 100),
      nn.LeakyReLU(),

      nn.LayerNorm(100),

      nn.Linear(100, 1),
      nn.Sigmoid()
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
  #----------------------------------------------------------------------------
    # simply run model
    return self.model(inputs)
    
    
  def train(self, inputs, targets):
  #----------------------------------------------------------------------------
    # calculate the output of the network
    outputs = self.forward(inputs)
        
    # calculate loss
    loss = self.loss_function(outputs, targets)

    # increase counter and accumulate error every 10
    self.counter += 1
    if (self.counter%10 == 0):
      self.progress.append(loss.item())
      pass
    if (self.counter%1000 == 0):
      print("counter = ", self.counter)
      pass

    # zero gradients, perform a backward pass, update weights
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  def plot_progress(self):
  #----------------------------------------------------------------------------
    df = pandas.DataFrame(self.progress, columns=['loss'])
    df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
    plt.show()
    pass
  pass

"""
# test discriminator can separate real data from random noise
D = Discriminator()

for image_data_tensor in celeba_dataset:
  # real data
  D.train(image_data_tensor, torch.FloatTensor([1.0]))
  # fake data
  D.train(generate_random_image((218,178,3)), torch.FloatTensor([0.0]))
  pass

#D.plot_progress()

# manually run discriminator to check it can tell real data from fake
for i in range(4):
  image_data_tensor = celeba_dataset[random.randint(0,20000)]
  print( D.forward( image_data_tensor ).item() )
  pass

for i in range(4):
  print( D.forward( generate_random_image((218,178,3))).item() )
  pass
"""

# generator class
class Generator(nn.Module):
#==============================================================================
  def __init__(self):
  #----------------------------------------------------------------------------
    # initialise parent pytorch class
    super().__init__()
        
    # define neural network layers
    self.model = nn.Sequential(
      nn.Linear(100, 3*10*10),
      nn.LeakyReLU(),

      nn.LayerNorm(3*10*10),

      nn.Linear(3*10*10, 3*218*178),

      nn.Sigmoid(),
      View((218,178,3))
    )

    # create optimiser, simple stochastic gradient descent
    self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    # counter and accumulator for progress
    self.counter = 0
    self.progress = []
    pass

  def forward(self, inputs):        
  #----------------------------------------------------------------------------
    # simply run model
    return self.model(inputs)

  def train(self, D, inputs, targets):
  #----------------------------------------------------------------------------
    # calculate the output of the network
    g_output = self.forward(inputs)

    # pass onto Discriminator
    d_output = D.forward(g_output)

    # calculate error
    loss = D.loss_function(d_output, targets)

    # increase counter and accumulate error every 10
    self.counter += 1
    if (self.counter % 10 == 0):
      self.progress.append(loss.item())
      pass

    # zero gradients, perform a backward pass, update weights
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    pass

  def plot_progress(self):
  #----------------------------------------------------------------------------
    df = pandas.DataFrame(self.progress, columns=['loss'])
    df.plot(ylim=(0),  figsize=(16,8), alpha=0.1, marker='.',
            grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0) )
    pass
  pass

"""
# check the generator output is of the right type and shape
G = Generator()
output = G.forward(generate_random_seed(100))

img = output.detach().numpy()
plt.imshow(img, interpolation='none', cmap='Blues')
plt.show()
"""
# create Discriminator and Generator

D = Discriminator()
G = Generator()

epochs = 1

for epoch in range(epochs):
  print ("epoch = ", epoch + 1)

  # train Discriminator and Generator

  for image_data_tensor in celeba_dataset:
    # train discriminator on true
    D.train(image_data_tensor, torch.FloatTensor([1.0]))
    
    # train discriminator on false
    # use detach() so gradients in G are not calculated
    D.train(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))
    
    # train generator
    G.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))
    pass
  pass

