#include<iostream>
#include<limits>

int main()
{ 
  float minv = std::numeric_limits<float>::epsilon();
  std::cout << minv << std::endl;
  return 0;
}

