
class A(object):
  def __init__(self, i):
    self.i = i
    print(f"__init__{self.i}")
    pass

  def __del__(self):
    print(f"desctruct {self.i} ...")
    pass

if __name__=="__main__":
  for i in range(5):
    a = A(i)
    del a # 显式调用比较好，可以消除隐患

  #print(" ---------------- ")
"""
  for j in range(5):
    with A(j) as b:
      print(f"b = A({j}) is using")
"""