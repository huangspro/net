#include "ConvolutionLayer.h"

ConvolutionLayer::ConvolutionLayer(int r, int c, int input_r, int input_c, int s):conkernel_row(r), conkernel_col(c), input_row(0), input_col(0), step(s){
  if((input_row-conkernel_row)%step!=0 || (input_col-conkernel_col)%step!=0){
    std::cout<< "Please resize the convolution kernel or step" << std::endl;
    exit(1);
  }else{
    //initialize the convolutional kernel
    static std::random_device rd;
    static std::default_random_engine generator(rd());
    std::normal_distribution < double > distribution(0.0, 0.141);
    for(int i=0;i<r*c;i++)conkernel.push_back(distribution(generator));
    //initialize the input and output variable
    for(int i=0;i<
  }
}

void ConvolutionLayer::forward(){
  double tem = 0;
  for(int i=0;i<output->size();i++){
    for(int ii=0;ii<(*output)[0].size();ii++){
      
      w(i, ii, );
    }
  }
}

void ConvolutionLayer::backward(){
  
}

double ConvolutionLayer::g(int row, int col){
  return (*input)[row][col];
}

void ConvolutionLayer::w(int row, int col, double data){
  (*output)[row][col] = data;
}

void ConvolutionLayer::g2(){
  
}
