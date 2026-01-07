#include "ConvolutionLayer.h"

ConvolutionLayer::ConvolutionLayer(int r, int c, int s):conkernel_row(r), conkernel_col(c), input_row(0), input_col(0), step(s){
  if((input_row+2*-conkernel_row)%step!=0 || (input_col+2*-conkernel_col)%step!=0){
    std::cout<< "Please resize the convolution kernel or step" << std::endl;
    exit(1);
  }else{
    output = new std::vector<std::vector<double>>((input_row+2*-conkernel_row)/step+1, std::vector<double>((input_col+2*-conkernel_col)/step+1));
  }
}

void ConvolutionLayer::forward(){
  
}

void ConvolutionLayer::backward(){
  
}

double ConvolutionLayer::g(int row, int col){
  return (*input)[row][col];
}

void ConvolutionLayer::w(int row, int col, double data){
  (*output)[row][col] = data;
}
