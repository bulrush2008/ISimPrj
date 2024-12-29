
from pathlib import Path

def assertFileExist(filePath:Path)->bool:
  return filePath.exists()