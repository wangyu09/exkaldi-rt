class A:
  def __init__(self):
    self.a = 10

class B(A):
  def __init__(self):
    super().__init__()
    self.b = 20

print( globals() )