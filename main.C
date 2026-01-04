#include "Node.h"
#include "Layer.h"
#include "Net.h"
#include<iostream>

using namespace std;

int main(){
  InputLayer* I=new InputLayer(2);
  NonlinearLayer* N=new NonlinearLayer(2, NonlinearLayer::TANH);
  MeanSquareErrorLayer* M=new MeanSquareErrorLayer(2);
  
  N->connect_to_last_layer_output(I->layer_output);
  N->connect_to_next_layer_input(M->input);
  
  vector<vector<double>> inputdata={15,25};
  vector<double> testdata={-0.1,-0.3};
  M->load_data_from_outside(testdata);
  I->input_data(inputdata);
  
  for(int i=0;i<10000;i++){
    I->forward();
    N->forward();
    M->forward();
    M->backward();
    N->backward();
    I->backward();
    I->train();
    if(i%100==0)cout<<"loss: "<<M->layer_output[0]->data<<endl;
  }

}
