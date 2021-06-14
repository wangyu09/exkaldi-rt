#include<iostream>
#include "feat/feature-functions.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[])
{ 
  using namespace kaldi;

  int order = atoi(argv[2]);
  int window = atoi(argv[4]);

  int num_frames, points;
  std::cin >> num_frames;
  std::cin >> points;
  std::cin.ignore();

  Matrix<BaseFloat> feats;
  feats.Resize(num_frames,points);
  feats.SetZero();

  for (int i=0; i<num_frames; i++)
  {
    for (int j=0; j<points; j++)
    {
      std::cin >> feats(i,j);
    }
  }

  DeltaFeaturesOptions dopts(order,window);
  Matrix<BaseFloat> new_feats;

  ComputeDeltas(dopts, feats, &new_feats);

  for (int m=0; m<new_feats.NumRows(); m++)
  {
    for (int n=0; n<new_feats.NumCols(); n++)
    {
      std::cout << new_feats(m,n) << " ";
    }
  }

  std::cout << std::endl;
  std::cout.flush();

  return 0;
}