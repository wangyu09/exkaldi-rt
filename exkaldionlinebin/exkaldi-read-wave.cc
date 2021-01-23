#include<iostream>
#include<vector>
#include<string>
#include"feat/wave-reader.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[])
{ 
  using namespace kaldi;

  std::string filename = argv[2];

  std::ifstream ifs;
  ifs.open( filename );
  
  WaveInfo header;
  header.Read(ifs);

  ifs.clear();
  ifs.seekg(0,std::ios::beg);

  WaveData reader;
  reader.Read(ifs);

  ifs.close();

  std::cout << header.SampFreq() << " ";
  std::cout << header.SampleCount() << " ";
  std::cout << header.NumChannels() << " ";
  std::cout << reader.Duration() << " ";

  Matrix<BaseFloat> data = reader.Data();

  for (int i = 0; i < data.NumCols(); ++i) 
  {
    for (int j = 0; j < data.NumRows(); ++j) 
    {
      std::cout << data(j, i) << " ";
    }
    std::cout << " ";
  }

  std::cout.flush();
  return 0;
}