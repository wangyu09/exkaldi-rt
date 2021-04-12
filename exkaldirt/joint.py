# coding=utf-8
#
# Yu Wang (University of Yamanashi)
# Apr, 2021
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

#from exkaldirt.base import Component, Joint, ENDPOINT, PIPE
#from exkaldirt.base import core_process_function, is_endpoint
from base import Component, Joint, ENDPOINT, PIPE, Packet
from base import is_endpoint
import copy

class Mapper(Component):
  '''
  Map packets with a filtering or transforamation function.
  Also its base class is Component.
  But it is used like a Joint object.
  '''
  def __init__(self,mapFunc,name=None):
    super().__init__(oKey="Null",name=name)
    assert callable(mapFunc)
    self.__map_function = mapFunc

  def core_loop(self):

    while True:

      action = self.decide_action()

      if action is False:
        break
      elif action is None:
        self.outPIPE.stop()
        break
      else:
        packet = self.get_packet()
        if is_endpoint(packet):
          self.put_packet( ENDPOINT )
          continue
        else:
          items = dict( packet.items() )
          items = self.__map_function( items )
          self.put_packet( Packet(items,cid=packet.cid,idmaker=packet.idmaker) )

class Spliter(Joint):
  '''
  Split packets (input form one PIPE) with a rule (then output to multiple PIPEs).
  '''
  def __init__(self,func,outNums,name=None):
    assert isinstance(outNums,int) and outNums > 1
    assert callable(func)
    self.__func = func
    super().__init__(self.__wrapped_function,outNums,name=name)
  
  def __wrapped_function(self,items):
    assert len(items) == 1
    return self.__func(items)

class Replicator(Joint):
  '''
  Split packets (input form one PIPE) by copy it to N copies (then output to multiple PIPEs).
  '''
  def __init__(self,outNums,name=None):
    assert isinstance(outNums,int) and outNums > 1
    super().__init__(self.__func,outNums,name=name)
  
  def __func(self,items):
    assert len(items) == 1
    return tuple(  copy.deepcopy(items[0]) for i in range(self.outNums) )

class Combiner(Joint):

  def __init__(self,func,name=None):
    assert callable(func)
    self.__comFunc = func
    super().__init__(self.__wrapped_function,outNums=1,name=name)
  
  def __wrapped_function(self,items):
    assert self.inNums > 1, f"{self.name}: inputs must more than 1"
    result = self.__comFunc( items )
    assert isinstance(result,dict)
    return result
  
class Merger(Joint):

  def __init__(self,name=None):
    super().__init__(self.__merge_function,outNums=1,name=name)
  
  def __merge_function(self,items):
    assert self.inNums > 1, f"{self.name}: inputs must more than 1"
    results = {}
    totalItems = 0
    for item in items:
      totalItems += len(item)
      results.update( item )
    assert totalItems == len(results), f"{self.name}: Multiple PIPE have the same item keys. This will cause data to be lost."
    return results