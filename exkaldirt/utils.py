
import numpy as np
from _io import BytesIO 
from multiprocessing import RawValue
import ctypes
import subprocess

def uint_to_bytes(value,length=4):
  return value.to_bytes(length=length,byteorder="little",signed=False)

def uint_from_bytes(value):
  return int.from_bytes(value,byteorder="little",signed=False)

def dtype_to_bytes(dtype):
  if dtype.name[0] == "i":
    flag = b"I"
  elif dtype.name[0] == "f":
    flag = b"F"
  else:
    raise Exception(f"Only signed int or float dtype can be convert to bytes using the utils.dtype_to_bytes function but the dtype is: {dtype.name}.")
  return  flag + uint_to_bytes(dtype.alignment,length=1)

def dtype_from_bytes(bdtype):
  flag = bdtype[0:1].decode()
  size = uint_from_bytes( bdtype[1:2] )
  if flag == "I":
    dtype = f"int{8*size}"
  elif flag == "F":
    dtype = f"float{8*size}"
  else:
    raise Exception(f"Unknown flag: {flag}")
  return dtype

def read_string(sp):
  out = ""
  while True:
    c = sp.read(1).decode()
    if c == " ":
      if out == "":
        continue
      else:
        break
    elif c == "":
      break
    else:
      out += c
  return out.strip()

def element_to_bytes(ele):
  assert isinstance(ele,(np.signedinteger,np.floating))
  dtype = dtype_to_bytes( ele.dtype )
  return dtype + ele.tobytes()
 
def element_from_bytes(ele):
  dtype = dtype_from_bytes( ele[0:2] )
  return np.frombuffer(ele[2:],dtype=dtype)[0]

def vector_to_bytes(vec):
  assert isinstance(vec,np.ndarray) and len(vec.shape) == 1
  bdtype = dtype_to_bytes( vec.dtype )
  return bdtype + vec.tobytes()

def vector_from_bytes(bvec):
  dtype = dtype_from_bytes( bvec[0:2] )
  return np.frombuffer(bvec[2:],dtype=dtype)

def matrix_to_bytes(mat):
  assert isinstance(mat,np.ndarray) and len(mat.shape) == 2
  bdtype = dtype_to_bytes( mat.dtype )
  frames = mat.shape[0]
  return bdtype + uint_to_bytes(frames,length=4) + mat.tobytes()

def matrix_from_bytes(mat):
  dtype = dtype_from_bytes( mat[0:2] )
  frames = uint_from_bytes( mat[2:6] )
  data = np.frombuffer( mat[6:], dtype=dtype )
  return data.reshape( [frames,-1] )

def cBool(value):
  return RawValue(ctypes.c_bool,value)

def cDouble(value):
  return RawValue(ctypes.c_double, value)

def cState(value):
  return RawValue(ctypes.c_ushort,value)

def cUint(value):
  return RawValue(ctypes.c_uint,value)

def encode_vector_temp(vec)->bytes:
  '''
  Define how to encode the vector data in order to send to subprocess.
  '''
  assert len(vec.shape) == 1 and 0 not in vec.shape
  return (" " + " ".join( map(str,vec)) + " ").encode()

def run_exkaldi_shell_command(cmd,inputs=None)->list:
  '''
  A simple function to run shell command.

  Args:
    _cmd_: a string of a shell command and its arguments.
    _inputs_: None or bytes object.
  '''
  assert isinstance(cmd,str) and len(cmd.strip()) > 0 
  
  if inputs is not None:
    assert isinstance(inputs,bytes),""
    stdin = subprocess.PIPE
  else:
    stdin = None

  p = subprocess.Popen(cmd,shell=True,stdin=stdin,stderr=subprocess.PIPE,stdout=subprocess.PIPE)
  (out,err) = p.communicate(input=inputs)
  cod = p.returncode

  if cod != 0:
    raise Exception(err.decode())
  else:
    return out.decode().strip().split()