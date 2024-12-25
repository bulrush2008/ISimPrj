
from pathlib import Path

def AssertFileExist(filePath:Path)->bool:
  return filePath.exists()