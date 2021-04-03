There are some examples based on mini_librispeech corpus.
If you want to run these programs, please prepare the following files:

1. final.mdl, HCLG.fst, words.txt

You can execute Kaldi's standard recipe to tri3b to get these files.

2. model.h5

A DNN model. 
We gave an example model structure using TensorFlow (2.4.0) in `neural_networks.py`.
You can use the MFCC feature and alignment can train it.

After preparing the above files, you can refer to or try to execute these programs.

1. 01_read_file.py

Read data stream from file and do online recognition.

2. 02_record_microphone_client.py, 02_record_microphone_server.py

Record data stream from microphone and do online recognition using a remote connection.
Please execute the programs on client and server respectively.
