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
  
  vector<vector<double>> inputdata={{1,1},{0,0},{0,1},{1,0}};
  vector<double> testdata={0,0,1,1};
  
  
  for(int i=0;i<10000;i++){
    for(int ii=0;ii<4;ii++){
      M->load_data_from_outside(testdata[ii]);
      I->input_data(inputdata[ii]);
      I->forward();
      N->forward();
      M->forward();
      M->backward();
      N->backward();
      I->backward();
      I->train();
    }
    if(i%100==0)cout<<"loss: "<<M->layer_output[0]->data<<endl;
  }

}
