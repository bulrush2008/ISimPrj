
class IdxList:
  def __init__(self):
    self.idxList = [ 1,   3,   5,   8,   11,  13,  15,  \
            18,  21,  23,  25,  28,  31,  34,  \
            37,  40,  43,  46,  49,  51,  53,  \
            55,  58,  61,  63,  65,  68,  71,  \
            73,  75,  78,  81,  84,  87,  90,  \
            93,  96,  99,  101, 103, 105, 108, \
            111, 113, 115, 118, 121, 123, 125]
    
    self.size = len(self.idxList)

  def __len__(self):
    return len(self.idxList)

  def __getitem__(self, idx):
    if idx >= self.size:
      raise IndexError(f"{idx} beyond range.")

    return self.idxList[idx]

if __name__=="__main__":

  idxList = IdxList()

  print(len(idxList))
  print(idxList[4]) # 11