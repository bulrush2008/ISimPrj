
import sys
import json

import h5py
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from Common.CaseSet import CaseSet
from Common.FSimDatasetPINN import FSimDatasetPINN
from Common.ModelPinn  import ModelPinn
from torch.utils.data import DataLoader
from tqdm import tqdm


from util.MetricTracker import MetricTracker




# import wandb
# wandb.login(key="1234")
# wandb.init(project="pinn_07_31")


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

    # data storing residuals between CFD field and prediction
    #   including both for train and test sets 
    # self.res_trn_hist = {}
    # self.res_tst_hist = {}

    self.test_Linf_tracker = MetricTracker()
    self.test_L2_tracker = MetricTracker()
    self.train_loss_tracker = MetricTracker()

    self.test_Linf_summary = []
    self.test_L2_summary = []
    self.train_loss_summary = []

    pass  # end '__init__()'

  def train(self):
  #-----------------------------------------------------------------------------
    # train the fields one has assigned, which must be in
    # ["P", "T", "U", "V", "W"]
    # the order in list does not matter

    print(f"*Fields Models Will Be Trained with Epochs {self.config['epochs']}.")

    trn_set = self.trn_set
    tst_set = self.tst_set

    file_path_h5 = self.file_path_h5

    # directory of loss png
    dir_png = Path("./Pics")
    if not dir_png.exists(): dir_png.mkdir(parents=True)

    # directory of model
    dir_model = Path(self.config["dict_dir_path"])
    if not dir_model.exists(): dir_model.mkdir(parents=True)

    # train
    self._train(
                train_set = trn_set,
                test_set  = tst_set,
                data_path = file_path_h5,
                dir_model = dir_model)
    pass  # end func 'self.train'
  

  def _train( self,
              train_set:list,
              test_set :list,
              data_path:Path,
              dir_model:Path )->None:
  #-----------------------------------------------------------------------------
    """
    Train the FnnPinn model by a give trainset, in which some cases field included.
    - train_set: list of case names in train set, each is a string
    - test_set : list of case names in test set, each is a string
    - data_path: path of data of train set
    """
  
    # train fields
    # obj to get the train data set
    # train set serves as (1) train & (2) error estimation
    model = ModelPinn(self.config)  # Create model first to get device
    
    fs_dataset_train = FSimDatasetPINN(data_path, train_set, self.config["var"])

    # obj to get the test data set
    # test set servers only as error estimation
    fs_dataset_test = FSimDatasetPINN(data_path, test_set, self.config["var"])

    
    train_loader = DataLoader(fs_dataset_train, batch_size=self.train_batch_size, shuffle=True)
    test_loader = DataLoader(fs_dataset_test, batch_size=self.train_batch_size, shuffle=True) 

    # train the model
    epochs = self.config["epochs"]

    for i in range(epochs):
      print(f" >> Training {self.config['var']}, epoch {i+1}/{epochs}")
      for params, coords, targets in tqdm(train_loader, desc=f"Epoch {i+1}/{epochs}"):
        batch_loss = model.train(params, coords, targets)
        self.train_loss_tracker.add(batch_loss)
        pass
      # wandb.log({"avg_train_loss": self.train_loss_tracker.average()})
      self.train_loss_summary.append(self.train_loss_tracker.summary())
      self.train_loss_tracker.reset()


      # # for the train set
      # e_train = []
      # for inp, field, _ in fs_dataset_train:
      #   e_train.append(model.calc_field_mse(inp, field))
      #   pass

      # for the test set  
      for params, coords, targets in test_loader:
        LInf_error = model.field_error(params, coords, targets, error_type="L-inf")
        L2_error = model.field_error(params, coords, targets, error_type="L-2")
        self.test_Linf_tracker.add(*LInf_error)
        self.test_L2_tracker.add(*L2_error)
        pass
      
      # wandb.log({"avg_test_Linf": self.test_Linf_tracker.average(), "avg_test_L2": self.test_L2_tracker.average()})
      self.test_Linf_summary.append(self.test_Linf_tracker.summary())
      self.test_L2_summary.append(self.test_L2_tracker.summary())
      self.test_Linf_tracker.reset()
      self.test_L2_tracker.reset()
      pass

    # save model parameters
    model_dicts_name = dir_model.joinpath(f"dict_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    torch.save(model.model.state_dict(), model_dicts_name)
    pass  

  
if __name__ == "__main__":
  fnn_pinn = FnnPinn()
  fnn_pinn.train()