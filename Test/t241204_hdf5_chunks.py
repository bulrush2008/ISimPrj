
import h5py
import numpy as np

data = np.random.rand(1000, 1000).astype("float32")

fileName = "optimized_data.h5"

with h5py.File(fileName, 'w') as f:
  f.create_dataset("dataset", data=data, chunks=(2,2))