#include<iostream>
#include<string>
#include<cstring>
#include"exkaldirtc/exkaldi-online-decoding.h"
#include"fstext/fstext-lib.h"

int main(int argc, char *argv[])
{ 
  using namespace exkaldi;
  using namespace std;

  LatticeFasterDecoderConfig dr_config;
  
  //configs
  //2 --beam 16.0 >0.
  KALDI_ASSERT(strcmp(argv[1],"--beam") == 0);
  dr_config.beam = atof(argv[2]);
  //4 --max-active > 1
  KALDI_ASSERT(strcmp(argv[3],"--max-active") == 0);
  dr_config.max_active = atoi(argv[4]);
  //6 --min-active 200 <max
  KALDI_ASSERT(strcmp(argv[5],"--min-active") == 0);
  dr_config.min_active = atoi(argv[6]);
  //8 --lattice-beam 10.0 > 0.
  KALDI_ASSERT(strcmp(argv[7],"--lattice-beam") == 0);
  dr_config.lattice_beam = atof(argv[8]);
  //10 --prune-interval 25 >0
  KALDI_ASSERT(strcmp(argv[9],"--prune-interval") == 0);
  dr_config.prune_interval = atoi(argv[10]);
  //determinize_lattice
  dr_config.determinize_lattice = true;
  //12 --beam-delta 0.5 >0.0
  KALDI_ASSERT(strcmp(argv[11],"--beam-delta") == 0);
  dr_config.beam_delta = atof(argv[12]);
  //14 --hash-ratio 2.0 >= 1.0
  KALDI_ASSERT(strcmp(argv[13],"--hash-ratio") == 0);
  dr_config.hash_ratio = atof(argv[14]);
  //16  --prune-scale 0.1 > 0. < 1.
  KALDI_ASSERT(strcmp(argv[15],"--prune-scale") == 0);
  dr_config.prune_scale = atof(argv[16]);

  ExkaldiDecodableOnlineOptions de_opts;
  //18 --acoustic-scale 0.1
  KALDI_ASSERT(strcmp(argv[17],"--acoustic-scale") == 0);
  de_opts.acoustic_scale = atof(argv[18]);
  //20 --lm-scale 1
  KALDI_ASSERT(strcmp(argv[19],"--lm-scale") == 0);
  BaseFloat lm_scale = atof(argv[20]);
  //22 --chunk-frames 64
  KALDI_ASSERT(strcmp(argv[21],"--chunk-frames") == 0);
  de_opts.chunk_frames = atoi(argv[22]);

  ExkaldiOnlineDecodingConfig config(dr_config,de_opts);

  //24 --allow-partial false
  KALDI_ASSERT(strcmp(argv[23],"--allow-partial") == 0);
  //bool allow_partial;
  //if (!strcmp(argv[28],"true")){ allow_partial = true; }
  //else { allow_partial = false; }
  //25 --n-bests 10
  KALDI_ASSERT(strcmp(argv[25],"--n-bests") == 0);
  int32 n_bests = atoi(argv[26]);
  //27 --silence-phones 1:2:3:4
  KALDI_ASSERT(strcmp(argv[27],"--silence-phones") == 0);
  std::string silence_phones = argv[28];
  //29 --frame-shift 0.01
  KALDI_ASSERT(strcmp(argv[29],"--frame-shift") == 0);
  BaseFloat frame_shift_in_seconds = atof(argv[30]);
  //32 --tmodel 
  KALDI_ASSERT(strcmp(argv[31],"--tmodel") == 0);
  std::string model_in_file = argv[32];
  //38 --fst
  KALDI_ASSERT(strcmp(argv[33],"--fst") == 0);
  std::string fst_in_str = argv[34];
  //40 --word-boundary
  KALDI_ASSERT(strcmp(argv[35],"--word-boundary") == 0);
  std::string word_boundary_file = argv[36];
  //42 --timeout
  KALDI_ASSERT(strcmp(argv[37],"--timeout") == 0);
  int32 TIMEOUT = atoi(argv[38]);
  //44 --timescale
  KALDI_ASSERT(strcmp(argv[39],"--timescale") == 0);
  int32 TIMESCALE = atoi(argv[40]);
  
  TransitionModel trans_model;
  ReadKaldiObject(model_in_file, &trans_model);

  fst::Fst<fst::StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);

  WordBoundaryInfo *word_boundary_info = NULL;
  if (word_boundary_file != "None")
  {
    WordBoundaryInfoNewOpts wdi_opts; // use default opts
    word_boundary_info = new WordBoundaryInfo(wdi_opts, word_boundary_file);
  }

  ExkaldiSingalUtteranceDecoder decoder(config,trans_model,*decode_fst);

  OnlineEndpointConfig ec_config;
  ec_config.silence_phones = silence_phones;

  while (true)
  { 
    ExkaldiSingalUtteranceDecoder decoder(config,trans_model,*decode_fst);

    while (true)
    { 
      // get chunk frames
      bool flag = decoder.RecieveFrames(TIMEOUT,TIMESCALE);
      // if received data
      if (flag)
      {
        decoder.AdvanceDecoding();
        if (decoder.IsLastDecoding()) {break;}
        if (decoder.IsEndpoint()) {break;}

        bool segover = decoder.EndpointDetected(ec_config,frame_shift_in_seconds);
        if (segover) { break;}

        Lattice lat;
        decoder.GetBestPath(false, &lat);

        EmitPartialResult(lat);
        std::cout << std::endl;
        std::cout.flush();      
      }
      else { break; }
  
    }

    if (decoder.IsTermination()) {
      std::cout << "-3 " << std::endl;
      std::cout.flush();
      break;
    }

    if ( decoder.NumFramesDecoded() == 0 ){
      std::cout << "-2 " << std::endl;
      std::cout.flush();
    }
    else {
      decoder.FinalizeDecoding();
      CompactLattice clat;
      decoder.GetLattice(true, &clat);

      EmitFinalResult(clat, de_opts.acoustic_scale, lm_scale, word_boundary_info,
                        trans_model, n_bests);
      
      std::cout << std::endl;
      std::cout.flush(); 
    }
  }

  if (word_boundary_file != "None"){
    delete word_boundary_info;
  }
  
  WaitForOver(TIMEOUT,TIMESCALE);

  return 0;
}
