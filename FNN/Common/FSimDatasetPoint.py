
from torch.utils.data import Dataset
from pathlib import Path

import torch
import numpy as np
import h5py


BLOCK_NUM = 8
SPACE_DIM = 3
PARAM_DIM = 3

class FSimDatasetPoint(Dataset):
#===============================================================================
  """
  - FSimDatasetPoint: Dataset for Point training on FSim data
  """
  def __init__(self, file:Path, case_list:list, var_name:list[str], mode:str="serial"):
  #-----------------------------------------------------------------------------
    """
    """
    self._mode = mode

    for v in var_name:
      if v not in ["P", "U", "V", "W", "T"]:
        raise ValueError("Error: the Variable Name must be P/U/V/W/T.")

    self.inp_serial   = []
    self.coord_serial   = []
    self.value_serial   = []

    self.inp_case = []
    self.axis_points_case = []
    self.coord_case = []
    self.value_case = []

    with h5py.File(file, 'r') as dataFile:
      for case in case_list:
        inp = dataFile[case]["input"]["inp"][:].astype(np.float32) # [PARAM_DIM]
        axis_points = {"x":[], "y":[], "z":[]}
        values = np.empty((0, len(var_name)), dtype=np.float32) # [0, len(var_name)] --> [CASE_POINT_NUM, len(var_name)]
        coords = np.empty((0, SPACE_DIM), dtype=np.float32) # [0, SPACE_DIM] --> [CASE_POINT_NUM, SPACE_DIM]
        for blk in range(BLOCK_NUM):
          key = f"block{blk:03d}"

          x = dataFile[case][key]["X"][:] # [x]
          y = dataFile[case][key]["Y"][:] # [y]
          z = dataFile[case][key]["Z"][:] # [z]

          axis_points["x"] += list(x) # List addition
          axis_points["y"] += list(y)
          axis_points["z"] += list(z)

          Z, Y, X = np.meshgrid(z, y, x, indexing='ij') # Each of X, Y, Z's shape: [x,y,z]
          coords_stack = np.stack([X, Y, Z], axis=-1).reshape(-1, SPACE_DIM) # [x*y*z, SPACE_DIM]
          coords = np.concatenate([coords, coords_stack], axis=0) # [coords[0] + x*y*z, SPACE_DIM]

          val = np.empty((x.shape[0]*y.shape[0]*z.shape[0], len(var_name)), dtype=np.float32) # [x*y*z, len(var_name)]
          for i, var in enumerate(var_name):
            var_val = dataFile[case][key][var][:].astype(np.float32) # [x*y*z,]
            val[:, i] = var_val
          values = np.concatenate([values, val], axis=0) # [values[0] + x*y*z, len(var_name)]

          inp_repeated = np.repeat(inp[np.newaxis, :], coords_stack.shape[0], axis=0) # [x*y*z, PARAM_DIM]

          self.inp_serial.append(inp_repeated)
          self.coord_serial.append(coords_stack)
          self.value_serial.append(val)

        self.inp_case.append(torch.tensor(inp, dtype=torch.float32))
        self.axis_points_case.append(axis_points)
        self.coord_case.append(torch.tensor(coords, dtype=torch.float32))
        self.value_case.append(torch.tensor(values, dtype=torch.float32))
    
    # [SUM(CASE_POINT_NUM_i) for i in range(len(case_list)), PARAM_DIM]
    self.inp_serial  = torch.cat([torch.from_numpy(arr) for arr in self.inp_serial], dim=0).to(torch.float32)
    # [SUM(CASE_POINT_NUM_i) for i in range(len(case_list)), SPACE_DIM]
    self.coord_serial  = torch.cat([torch.from_numpy(arr) for arr in self.coord_serial], dim=0).to(torch.float32)
    # [SUM(CASE_POINT_NUM_i) for i in range(len(case_list)), len(var_name)]
    self.value_serial  = torch.cat([torch.from_numpy(arr) for arr in self.value_serial], dim=0).to(torch.float32)

    assert self.inp_serial.shape[0] == self.coord_serial.shape[0] == self.value_serial.shape[0]
    
  def __len__(self):
    if self.mode == "serial":
      return self.inp_serial.shape[0]
    elif self.mode == "case":
      return len(self.inp_case)
    else:
      raise ValueError(f"Invalid mode: {self.mode}")

  def __getitem__(self, idx):
    if self.mode == "serial":
      return (
        self.inp_serial[idx],
        self.coord_serial[idx],
        self.value_serial[idx]
      )
    elif self.mode == "case":
      return (
        self.inp_case[idx],
        self.axis_points_case[idx],
        self.coord_case[idx],
        self.value_case[idx]
      )
    else:
      raise ValueError(f"Invalid mode: {self.mode}")
  
  @property
  def mode(self):
    return self._mode
  
  @mode.setter
  def mode(self, val):
    if val not in ["serial", "case"]:
      raise ValueError(f"Invalid mode: {val}")
    self._mode = val
    
  
if __name__ == "__main__":
  from CaseSet import CaseSet
  caseSet = CaseSet(ratio = 0.2)
  trnSet, tstSet = caseSet.splitSet()
  import time 
  start_time = time.time()
  fsDataset = FSimDatasetPoint(Path("../FSCases/FSHDF/MatrixData.h5"), trnSet, ["U", "V", "W"])
  print(len(fsDataset)) 
  end_time = time.time()
  print(f"Time taken: {end_time - start_time} seconds")
  from torch.utils.data import DataLoader
  train_loader = DataLoader(fsDataset, batch_size=3, shuffle=True)
  for batch in train_loader:
    inputs, coords, values = batch
    print(f"Inputs shape: {inputs.shape}, Coords shape: {coords.shape}, Values shape: {values.shape}")
    print("First input:", inputs[0])
    print("First coords:", coords[0])
    print("First value:", values[0])

