import torch
import torch.nn as nn
import math
import pandas
import h5py
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

from .PinnLoss import incompressibility_loss

class ModelPoint(nn.Module):
#===============================================================================
  """
  - ModelPoint class: Core of FnnPoint
  - 方法类
  """
  # initialize PyTorch pararent class
  def __init__(self, config:dict):
  #-----------------------------------------------------------------------------
    super().__init__()
    self.load_dict = config.get("load_dict", False) # 是否载入之前训练的模型
    self.dropout = config.get("dropout", None)
    self.alpha = config.get("alpha", 0.5)
    self.learning_rate = config.get("learning_rate", 0.005)
    self.activation = config.get("activation", "leaky_relu")
    if torch.cuda.is_available():
        self.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        self.device = torch.device("mps")
    else:
        self.device = torch.device("cpu")
    
    self.dict_path = Path(config["dict_dir_path"]).joinpath(f"{config['dict_load_name']}")

    print(f"loading model from {self.dict_path}")

    print(f"*Use device: {self.device}")
    print(f"*Use dropout: {self.dropout}")
    print(f"*Use learning rate: {self.learning_rate}")
    print(f"*Use activation: {self.activation}")
    print(f"*Load Previous Model: {'yes' if self.load_dict else 'no'}")
    
    # Additional GPU information
    if torch.cuda.is_available():
        print(f"*CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"*CUDA Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Build model from configurable architecture
    self.model = self._build_model(config["architecture"])

    print("*Model structure:")
    print(self.model)
    
    self.model = self.model.to(self.device)

    if self.load_dict:
      self.model.load_state_dict(torch.load(self.dict_path, map_location=self.device))
      print(f"Load model from {self.dict_path}")
    else:
      # initialize weights，using He Kaiming method now
      self._initialize_weights()
      print(f"Initialize model weights")
      pass

    # regressive problem, MSE is proper
    self.loss_function = nn.MSELoss()
    self.pinn_loss_function = incompressibility_loss
    self.optimiser = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    pass

  def _build_model(self, architecture):
  #-----------------------------------------------------------------------------
    """
    Build neural network model from architecture configuration
    - architecture: list of [input_size, output_size] pairs
    """
    layers = []
    
    for i, (in_features, out_features) in enumerate(architecture):
        layers.append(nn.Linear(in_features, out_features))
        
        if i < len(architecture) - 1:
            if self.activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.02))
            elif self.activation == "relu":
                layers.append(nn.ReLU())
            elif self.activation == "tanh":
                layers.append(nn.Tanh())
            elif self.activation == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.LeakyReLU(0.02))
                
            if self.dropout is not None and self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))

            layers.append(nn.LayerNorm(out_features))
    
    return nn.Sequential(*layers)

  def _initialize_weights(self):
  #-----------------------------------------------------------------------------
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
      pass
    pass

  def forward(self, params, coords):
  #-----------------------------------------------------------------------------
    # inputs = inputs.to(self.device)
    inputs = torch.cat([params, coords], dim=1) # (N, 6)
    inputs = inputs.to(self.device)
    return self.model(inputs)

  def train_step(self, params, coords, targets):
  #-----------------------------------------------------------------------------
    """
    - 神经网络，根据输入和标签，进行训练

    - params : 控制参数
    - targets: 目标值
    """
    params = params.to(self.device)
    coords = coords.to(self.device).requires_grad_(True)  # Enable gradients for PINN
    targets = targets.to(self.device)
    
    outputs = self.forward(params, coords) # (N, len(var_name))
    data_loss = self.loss_function(outputs, targets)
    # pinn_loss = self.pinn_loss_function(outputs, coords)
    # loss = (1 - self.alpha) * data_loss + self.alpha * pinn_loss
    loss = data_loss
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    return loss.item()
  
  
  def field_error(self, params, coords, targets, error_type="L-inf"):
  #-----------------------------------------------------------------------------
    """
    Calculate the error of each case between prediction and real.
    Function's three input parameters are same with 'self.train(...)'

    - params: input parameters
    - coords: coordinates
    - targets: data of the real field of each case
    """
    # Move inputs and targets to device
    params = params.to(self.device)
    coords = coords.to(self.device)
    targets = targets.to(self.device)

    # Set the model to eval mode so that dropout and batch norm behave correctly
    self.model.eval()

    inputs = torch.cat([params, coords], dim=1) # (N, 6)
    output = self.forward(inputs).detach().cpu().numpy()

    # a/b both of type torch.FloatTensor
    a = output; b = targets.detach().cpu().numpy()
    #e = sum(abs(a-b))
    if error_type == "L-inf":
      e = np.max(np.abs(a-b), axis=1)
    elif error_type == "L-2":
      e = np.sqrt(np.mean((a-b)**2, axis=1))
    else:
      raise ValueError(f"Error: {error_type} error not implemented.")
    return e