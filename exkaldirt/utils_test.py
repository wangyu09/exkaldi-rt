import utils
import numpy as np
#from exkaldirt import utils
from _io import BytesIO

######################################
# uint_to_bytes, uint_from_bytes
######################################

a = 16
b = utils.uint_to_bytes(16,1) 
print( a, "->", b )

c = utils.uint_from_bytes( b )
print( b, "->", c )

######################################
# dtype_to_bytes, dtype_from_bytes
######################################

a = np.int16(16).dtype
b = utils.dtype_to_bytes( a )
print( a, "->", b )

c = utils.dtype_from_bytes( b )
print( b, "->", c )

######################################
# read_string
######################################

a = b" test1  test2"
with BytesIO(a) as sp:
  print( "first time",
        utils.read_string(sp)
      )
  print( "second time",
        utils.read_string(sp)
      )
  print( "third time",
        utils.read_string(sp)
      )    

######################################
# element_to_bytes,element_from_bytes
######################################

a = np.int16(16)
b = utils.element_to_bytes( a )
print( a, "->", b )

c = utils.element_from_bytes( b )
print( b, "->", c )

######################################
# vector_to_bytes,vector_from_bytes
######################################

a = np.ones([5,],dtype="float32")
b = utils.vector_to_bytes( a )
print( a, "->", b )

c = utils.vector_from_bytes( b )
print( b, "->", c )

######################################
# cBool, cDouble, cState
######################################

print( utils.cBool(True) )
print( utils.cDouble(0.0) )
print( utils.cState(1) )