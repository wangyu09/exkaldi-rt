from __future__ import absolute_import

# import the common modules
from exkaldi2 import version
from exkaldi2 import base
from exkaldi2.base import info
from exkaldi2 import stream
from exkaldi2 import transmit

# these modules will be hidden if exkaldi2 only run on local environment without Kaldi
if info.KALDI_EXISTED:
  from exkaldi2 import feature
  from exkaldi2 import decode

