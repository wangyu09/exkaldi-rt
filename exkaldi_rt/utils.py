# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# May, 2021
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess

def declare(condition, emessage=None, objName=None):
  if emessage is None:
    emessage = condition
    condition = True

  assert isinstance(emessage,str)
    
  if condition:
    if objname is None:
      raise Exception(emessage)
    else:
      assert isinstance(objName,str)
      raise Exception(f"{objName}: {emessage}")

def run_exkaldi_shell_command(cmd,inputs=None)->list:
  '''
  A simple function to run shell command.

  Args:
    _cmd_: a string of a shell command and its arguments.
    _inputs_: None or bytes object.
  '''
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
    out =  out.decode().strip().split()
    return out