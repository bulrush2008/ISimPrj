
class X(object):
  def __init__(self):
    pass

  def __call__(self, t):
    return t*t
  pass

if __name__ == "__main__":
  x = X()

  y = x(2.5)  # 6.25?
  print(y)