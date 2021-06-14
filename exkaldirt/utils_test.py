import utils
import numpy as np
from _io import BytesIO

######################################
# uint_to_bytes, uint_from_bytes
######################################

def test_uint_float():
  a = 16
  b = utils.uint_to_bytes(a,length=1) 
  print( a, "->", b )

  c = utils.uint_from_bytes( b )
  print( b, "->", c )

  d = 0.0001
  e = utils.float_to_bytes(d)
  print( d, "->", e )

  f = utils.float_from_bytes(e)
  print( e, "->", f)

test_uint_float()

######################################
# dtype_to_bytes, dtype_from_bytes
######################################

def test_dtype():
  a = np.int16(16).dtype
  b = utils.dtype_to_bytes( a )
  print( a, "->", b )

  c = utils.dtype_from_bytes( b )
  print( b, "->", c )

test_dtype()

######################################
# read_string
######################################

def test_read_string():

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

test_read_string()

######################################
# element_to_bytes,element_from_bytes
######################################

def element_test():

  a = np.int16(16)
  b = utils.element_to_bytes( a )
  print( a, "->", b )

  c = utils.element_from_bytes( b )
  print( b, "->", c )

element_test()

######################################
# vector_to_bytes,vector_from_bytes
######################################

def vector_test():

  a = np.ones([5,],dtype="float32")
  b = utils.vector_to_bytes( a )
  print( a, "->", b )

  c = utils.vector_from_bytes( b )
  print( b, "->", c )

vector_test()

######################################
# matrix_to_bytes,matrix_from_bytes
######################################

def matrix_test():

  a = np.ones([2,5],dtype="float32")
  b = utils.matrix_to_bytes( a )
  print( a, "->", b )

  c = utils.matrix_from_bytes( b )
  print( b, "->", c )

matrix_test()