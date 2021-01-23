#include<iostream>
#include<vector>

#include "base/kaldi-math.h"
//#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[])
{ 
  using namespace kaldi;

  int factor = atoi(argv[2]);

  int frames, points;
  std::cin >> frames;
  std::cin >> points;

  //Matrix<BaseFloat> wave;
  //wave.Resize(frames,points);

  RandomState rstate;
  float temp;
  for (int i=0; i<frames; i++)
  { 
    for (int j=0; j<points; j++)
    {
      std::cin >> temp;
      std::cout << temp + RandGauss(&rstate) * factor << " ";
    }
  }
  /*
  for (int i=0; i<num_frames; i++)
  {
    std::cout << wave[i] << " ";
  }
  */
  std::cout << std::endl;
  std::cout.flush();

  return 0;
}