#include "ConvolutionLayer.h"

ConvolutionLayer::ConvolutionLayer(int r, int c, int input_r, int input_c, int s):conkernel_row(r), conkernel_col(c), input_row(input_r), input_col(input_c), step(s){
  if((input_r-r)%s!=0 || (input_c-c)%s!=0){
    std::cout<< "Please resize the convolution kernel or step" << std::endl;
    exit(1);
  }else{
    //initialize the convolutional kernel
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution < double > distribution(0.0, 0.141);
    for(int i=0;i<r;i++){
      std::vector<Var*> temm;
      for(int j=0;j<c;j++){
        Var* tem = new Var(distribution(generator),0);
        temm.push_back(tem);
      }
      conkernel.push_back(temm);
    }
    //initialize the input and output variable
    //input
    for(int i=0;i<input_r;i++){
      std::vector<Var*> tem;
      for(int ii=0;ii<input_c;ii++){
        tem.push_back(new Var(0,0));
      }
      input.push_back(tem);
    }
    //output, firstly calculate the output size
    for(int i=0;i<(input_row-conkernel_row)/step+1;i++){
      std::vector<Var*> tem;
      for(int ii=0;ii<(input_col-conkernel_col)/step+1;ii++){
        tem.push_back(new Var(0,0));
      }
      layer_output.push_back(tem);
    }
  }
}
//pass data forward
void ConvolutionLayer::forward(){
  double output_i = 0;
  double output_j = 0;
  for(int i=0;i<(input_row-conkernel_row)/step+1;i+=step){
    for(int j=0;j<(input_col-conkernel_col)/step+1;j+=step){
      //calculate the output on point i,j
      w(output_i, output_j, cal(i,j));
      output_j++;
    }
    output_j=0;
    output_i++;
  }
}
//pass gradient backward, according to the formula: Output_i,j/dK_k1,k2 = I_i+k1,j+k2
void ConvolutionLayer::backward(){
  for(int k1=0;k1<conkernel.size();k1++){//for each kernel
    for(int k2=0;k2<conkernel.size();k2++){
      for(int i=0;i<layer_output.size();i++){//for each output
        for(int j=0;j<layer_output[0].size();j++){
          //gradient++
          g2(k1,k2)->gradient+=g(i+k1,j+k2)*layer_output[i][j]->gradient;
          std::cout<<"test: "<<g2(k1,k2)->gradient<<std::endl;
        }
      }
    }
  }
}
//get data from input
double ConvolutionLayer::g(int row, int col){
  return input[row][col]->data;
}
//write data to output
void ConvolutionLayer::w(int row, int col, double data){
  conkernel[row][col]->data = data;
}
//get data from kernel
Var* ConvolutionLayer::g2(int row, int col){
  return input[row][col];
}
//calculate out the output of input, input the left-top
double ConvolutionLayer::cal(int a, int b){
  double result=0;
  for(int i=0;i<conkernel_row;i++){
    for(int ii=0;ii<conkernel_col;ii++){
      result+=g(a+i,b+ii)*g2(i,ii)->data;
    }
  }
  return result;
}

void ConvolutionLayer::load_data_from_outside(std::vector<std::vector<double>>& input_data){
  for(int i=0;i<input_row;i++){
    for(int ii=0;ii<input_col;ii++){
      input[i][ii]->data = input_data[i][ii];
    }
  }
}

void ConvolutionLayer::connect_to_next_conlayer_input(std::vector<std::vector<Var*>> next_layer_input){
  for(int i = 0; i < layer_output.size(); i++) {
    for(int j = 0; j < layer_output[0].size(); j++) {
      delete layer_output[i][j];
      layer_output[i][j] = next_layer_input[i*j];
    }
  }
}

void ConvolutionLayer::connect_to_last_conlayer_output(std::vector<std::vector<Var*>> last_layer_output){
  for (int i = 0; i < last_layer_output.size(); i++) {
    delete input[i];
    input[i] = last_layer_output[i];
  }
}

void ConvolutionLayer::connect_to_next_layer_input(std::vector<std::vector<Var*>> next_layer_input){
  for(int i = 0; i < next_layer_input.size(); i++) {
    delete layer_output[i];
    layer_output[i] = next_layer_input[i];
  }
}

void ConvolutionLayer::connect_to_last_layer_output(std::vector<std::vector<Var*>> last_layer_output){
  for (int i = 0; i < last_layer_output.size(); i++) {
    delete input[i];
    input[i] = last_layer_output[i];
  }
}
