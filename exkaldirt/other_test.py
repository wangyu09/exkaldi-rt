from collections import namedtuple

A = namedtuple("T",["t1","t2"])(0,1)
print( A._asdict().keys() )