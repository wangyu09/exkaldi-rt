import exkaldirt

#### info
info = exkaldirt.base.info
print( "Kaldi Existed:", info.KALDI_EXISTED )
print( "Command Root:", info.CMDROOT )

#### PIPE

pipe = exkaldirt.base.PIPE

for i in range(10):
  pipe.put( exkaldirt.base.Element(i) )

print( "PIPE size:",pipe.size() )
print( pipe.to_list() )
