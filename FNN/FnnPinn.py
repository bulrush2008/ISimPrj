
import sys
import json

import h5py
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from Common.CaseSet import CaseSet
from Common.FSimDatasetPinn import FSimDatasetPinn
from Common.ModelPinn  import ModelPinn
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.MetricTracker import MetricTracker
from util.Write2HDF import write2HDF
from util.error import LInfError, L2Error, L1Error
import wandb


class FnnPinn(object):
#===============================================================================
  """
  - 应用任务类

  - 调用方法类和数据类，实现特定的应用任务
  """
  def __init__( self ):
  #-----------------------------------------------------------------------------
    # split the cases into train and test sets
    # now: 125 = 100 + 25
    with open("./fnnPinn.json", 'r') as inp:
      self.config = json.load(inp)
      pass

    self.train_batch_size = self.config["train_batch_size"]
    case_set = CaseSet(ratio=self.config["test_ratio"])
    trn_set, tst_set = case_set.splitSet()
    self.trn_set = trn_set
    self.tst_set = tst_set

    print(f"*Use trainBatchSize: {self.train_batch_size}")

    # path of data used as training and possibly test
    self.file_path_h5 = Path(self.config["train_data"])

    self.fs_dataset_train = FSimDatasetPinn(file=self.file_path_h5, caseList=trn_set, varName=self.config["var"])
    self.fs_dataset_test = FSimDatasetPinn(file=self.file_path_h5, caseList=tst_set, varName=self.config["var"])
    self.train_loader = DataLoader(self.fs_dataset_train, batch_size=self.train_batch_size, shuffle=True, 
                                   num_workers=4, pin_memory=True)
    self.test_loader = DataLoader(self.fs_dataset_test, batch_size=self.train_batch_size, shuffle=True, 
                                  num_workers=4, pin_memory=True) 
    
    self.model = ModelPinn(self.config)  # Create model first to get device

    pass  # end '__init__()'

  def train(self):
  #-----------------------------------------------------------------------------
    # train the fields one has assigned, which must be in
    # ["P", "T", "U", "V", "W"]
    # the order in list does not matter

    print(f"*Fields Models Will Be Trained with Epochs {self.config['epochs']}.")

    self.test_Linf_tracker = MetricTracker()
    self.test_L2_tracker = MetricTracker()
    self.train_loss_tracker = MetricTracker()

    self.test_Linf_summary = []
    self.test_L2_summary = []
    self.train_loss_summary = []


    # directory of loss png
    dir_png = Path("./Pics")
    if not dir_png.exists(): dir_png.mkdir(parents=True)

    # directory of model
    dir_model = Path(self.config["dict_dir_path"])
    if not dir_model.exists(): dir_model.mkdir(parents=True)

    # train
    self._train(dir_model = dir_model)
    pass  # end func 'self.train'
  
  def evaluate(self, write_vtk:bool=False):
    """
    """
    self.model.eval()
    self.fs_dataset_test.mode = "case"
    error = {"L-inf": [], "L-2": [], "L-1": []}

    for inp, axis_points, coord, value in tqdm(self.fs_dataset_test, desc="Evaluating"):
      with torch.no_grad():
        inp = inp.repeat(coord.shape[0], 1)
        pred = self.model.forward(params=inp, coords=coord)
        pred = pred.reshape(1, pred.shape[0]) # (1, P)
        if write_vtk:
          path = Path(self.config["eval_dir_path"]).joinpath(f"{inp[0][0]}_{inp[0][1]}_{inp[0][2]}.h5")
          write2HDF(pred, path, self.config["var"], axis_points)
        error["L-inf"].append(LInfError(pred, value))
        error["L-2"].append(L2Error(pred, value))
        error["L-1"].append(L1Error(pred, value))
        pass
      pass
    self.model.train()
    return error
      

  def _train( self,
              dir_model:Path )->None:
  #-----------------------------------------------------------------------------
    """
    Train the FnnPinn model by a give trainset, in which some cases field included.
    - train_set: list of case names in train set, each is a string
    - test_set : list of case names in test set, each is a string
    - data_path: path of data of train set
    """
    epochs = self.config["epochs"]

    for i in range(epochs):
      print(f" >> Training {self.config['var']}, epoch {i+1}/{epochs}")
      for params, coords, targets in tqdm(self.train_loader, desc=f"Epoch {i+1}/{epochs}"):
        batch_loss = self.model.train_step(params, coords, targets)
        self.train_loss_tracker.add(batch_loss)
        # print("Memory allocated:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
        # print("Memory reserved:", torch.cuda.memory_reserved(0) / 1024**2, "MB")
        pass
      wandb.log({"avg_train_loss": self.train_loss_tracker.average()})
      self.train_loss_summary.append(self.train_loss_tracker.summary())
      self.train_loss_tracker.reset()

      # # for the test set  
      # for params, coords, targets in self.test_loader:
      #   LInf_error = self.model.field_error(params, coords, targets, error_type="L-inf")
      #   L2_error = self.model.field_error(params, coords, targets, error_type="L-2")
      #   self.test_Linf_tracker.add(*LInf_error)
      #   self.test_L2_tracker.add(*L2_error)
      #   pass
      test_error = self.evaluate()
      self.test_Linf_tracker.add(*test_error["L-inf"])
      self.test_L2_tracker.add(*test_error["L-2"])

      
      wandb.log({"avg_test_Linf": self.test_Linf_tracker.average(), "avg_test_L2": self.test_L2_tracker.average()})
      self.test_Linf_summary.append(self.test_Linf_tracker.summary())
      self.test_L2_summary.append(self.test_L2_tracker.summary())
      self.test_Linf_tracker.reset()
      self.test_L2_tracker.reset()
      pass

    # save model parameters
    model_dicts_name = dir_model.joinpath(f"dict_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    torch.save(self.model.state_dict(), model_dicts_name)
    pass  

  
if __name__ == "__main__":
  # Read wandb key from file
  try:
      with open("../wandb.key", 'r') as f:
          wandb_key = f.read().strip()
  except FileNotFoundError:
      raise FileNotFoundError("wandb.key file not found. Please create a file named 'wandb.key' containing your wandb API key.")

  wandb.login(key=wandb_key)
  wandb.init(project="pinn_08_04")
  fnn_pinn = FnnPinn()
  fnn_pinn.train()
  # e = fnn_pinn.evaluate()
  # print(e)