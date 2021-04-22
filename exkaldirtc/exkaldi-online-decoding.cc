
#include <iostream>
#include <ctime>
#include <string>
#include"base/kaldi-common.h"
#include"exkaldi-online-decoding.h"

namespace exkaldi {

using namespace kaldi;
using namespace std;

void TimeDelay(int msec)
{
  clock_t now = clock();
  while ( (clock()-now)/CLOCKS_PER_MSEC < msec ) {};
}

ExkaldiDecodableOnline::ExkaldiDecodableOnline(
  TransitionModel &trans_model,
  const ExkaldiDecodableOnlineOptions &opts):
  opts_(opts),
  trans_model_(trans_model),
  begin_frame_(0),
  last_frame_id_(-1),
  frames_ready_(0),
  available_frames_(0)
{ 
  pdf_ids_ = trans_model_.NumPdfs();
  loglikes_.Resize(opts_.chunk_frames, pdf_ids_);
  loglikes_.SetZero();
}

BaseFloat ExkaldiDecodableOnline::LogLikelihood(int32 frame, int32 index)
{
  KALDI_ASSERT(frame >= begin_frame_ && frame < begin_frame_ + available_frames_);
  int32 pdf_id = trans_model_.TransitionIdToPdf(index);
  //return opts_.acoustic_scale * loglikes_(frame-begin_frame_,pdf_id);
  return loglikes_(frame-begin_frame_,pdf_id); // We will scale the acoustic prob in python program but not here.
}

bool ExkaldiDecodableOnline::IsLastFrame(int32 frame) const
{
  return (frame == last_frame_id_);
}

int32 ExkaldiDecodableOnline::NumFramesReady() const
{
  return frames_ready_;
}

bool ExkaldiDecodableOnline::RecieveFrames(const int timeout, const int timescale)
{ 
  if (ArrivedLastChunk())
  {
    return false;
  }

  int counter = 0;

  while (true)
  {
    if (cin.peek() == EOF)
    {
      TimeDelay(timescale);
      counter += timescale;
      if (counter >= timeout)
        {KALDI_ERR << "Timeout: input stream did not receive any data.";}
    }
    else if (cin.peek() == ' ' || cin.peek() == '\n')
    { 
      cin.ignore();
    }
    else
    { 
      begin_frame_ += available_frames_; // Update the begin ID with previous number of frames.

      int flag;
      int frames;

      //Read Header
      cin >> flag;
      //std::cout << "read flag done:" << flag << std::endl;
      if (flag == -3){
        Terminate();
        return false;
      }
      else if (flag == -2){
        //SetEndpoint();
        //last_frame_id_ = frames_ready_;
        //return false;
        cin >> frames;
        if (frames > 0){
          if ( !( frames >=0 && frames <= opts_.chunk_frames )) 
          KALDI_ASSERT( frames >=0 && frames <= opts_.chunk_frames);
          for (int32 i=0; i<frames; i++){
            for (int32 j=0; j<pdf_ids_; j++){
              cin >> loglikes_(i,j);
            }
          }
          frames_ready_ += frames;
          available_frames_ = frames;

          SetEndpoint();
          last_frame_id_ = frames_ready_;
          return true;
        } else {
          SetEndpoint();
          last_frame_id_ = frames_ready_;
          return false;          
        }
      }
      else if (flag == -1){

        cin >> frames;
        if ( !( frames >0 && frames <= opts_.chunk_frames )) 
        KALDI_ASSERT( frames >0 && frames <= opts_.chunk_frames);

        for (int32 i=0; i<frames; i++){
          for (int32 j=0; j<pdf_ids_; j++){
            cin >> loglikes_(i,j);
          }
        }

        frames_ready_ += frames;
        available_frames_ = frames;

        break;
      }
      else { KALDI_ERR << "Flag must be (-1 -> activity, -2 -> endpoint, -3 -> termination) but got known value:" << flag;}
    }
  }

  return true;
}

ExkaldiSingalUtteranceDecoder::ExkaldiSingalUtteranceDecoder(
    const ExkaldiOnlineDecodingConfig &config,
    TransitionModel &tmodel,
    fst::Fst<fst::StdArc> &fst):
    config_(config),
    tmodel_(tmodel),
    decodable_(tmodel, config.decodable_opts),
    decoder_(fst, config.decoder_opts) 
{
  decoder_.InitDecoding();
}

void ExkaldiSingalUtteranceDecoder::AdvanceDecoding() {
  decoder_.AdvanceDecoding(&decodable_);
}

void ExkaldiSingalUtteranceDecoder::FinalizeDecoding() {
  decoder_.FinalizeDecoding();
}

int32 ExkaldiSingalUtteranceDecoder::NumFramesDecoded() const {
  return decoder_.NumFramesDecoded();
}

void ExkaldiSingalUtteranceDecoder::GetLattice(bool end_of_utterance,
                                             CompactLattice *clat) const 
{
  if (NumFramesDecoded() == 0)
    KALDI_ERR << "You cannot get a lattice if you decoded no frames.";
  Lattice raw_lat;
  decoder_.GetRawLattice(&raw_lat, end_of_utterance);

  if (!config_.decoder_opts.determinize_lattice)
    KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

  BaseFloat lat_beam = config_.decoder_opts.lattice_beam;
  DeterminizeLatticePhonePrunedWrapper(
      tmodel_, &raw_lat, lat_beam, clat, config_.decoder_opts.det_opts);
}

void ExkaldiSingalUtteranceDecoder::GetBestPath(bool end_of_utterance,
                                              Lattice *best_path) const 
{
  decoder_.GetBestPath(best_path, end_of_utterance);
}

bool ExkaldiSingalUtteranceDecoder::EndpointDetected(
                  const OnlineEndpointConfig &config, 
                  BaseFloat frame_shift_in_seconds) 
{
    return kaldi::EndpointDetected(config, 
                                   tmodel_,
                                   frame_shift_in_seconds,
                                   decoder_);  
}

int EmitPartialResult(const Lattice &lat)
{
  std::vector<int32> words;
  std::vector<int32> alignment;
  LatticeWeight weight;
  
  GetLinearSymbolSequence(lat, &alignment, &words, &weight);
  std::cout << "-1 ";
  for (int32 i=0; i<words.size(); i++)
  {
    std::cout << words[i] << " ";
  }
  return 0;
}

int EmitFinalResult(CompactLattice &clat, 
                    BaseFloat acwt, 
                    BaseFloat lmwt, 
                    WordBoundaryInfo *wbi,
                    TransitionModel &trans_model,
                    int32 N_best)
{ 
  // Scale lattice
  //if (acwt) { 
  //  BaseFloat inv_acoustic_scale = 1.0/acwt;
  //  fst::ScaleLattice(fst::AcousticLatticeScale(inv_acoustic_scale), &clat);
  //  }
  fst::ScaleLattice(fst::LatticeScale(lmwt, 1.0), &clat);
  
  //Apply word boundary
  if (wbi != nullptr){
    CompactLattice aligned_clat;
    if (WordAlignLattice(clat, trans_model, *wbi, 0, &aligned_clat)) {
      clat = aligned_clat;
    }
  }
  Lattice lat;
  ConvertLattice(clat, &lat);

  std::vector<Lattice> nbest_lats; 
  Lattice nbest_lat;
  fst::ShortestPath(lat, &nbest_lat, N_best);
  fst::ConvertNbestToVector(nbest_lat, &nbest_lats);

  std::cout << "-2 ";
  for (int32 i=0; i<nbest_lats.size(); i++ )
  {
    EmitPartialResult(nbest_lats[i]);
  }

  return 0;
}

void WaitForOver(int32 timeout, int32 timescale){ 
  int counter = 0;
  while (true){
    if (cin.peek() == EOF){
        TimeDelay(timescale);
        counter += timescale;
        if (counter >= timeout)
          {KALDI_ERR << "Timeout: input stream did not receive any singal.";}
    }
    else if (cin.peek() == ' ' || cin.peek() == '\n'){ 
      cin.ignore();
    }
    else{
      string singal;
      cin >> singal;
      if (singal == "over"){
        break;
      }
    }
  }
}

}