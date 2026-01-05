#include "Node.h"
#include "Layer.h"
#include<iostream>

using namespace std;

int main(){
  InputLayer* I=new InputLayer(2);
  NonlinearLayer* N1=new NonlinearLayer(2, NonlinearLayer::TANH);
  MeanSquareErrorLayer* M=new MeanSquareErrorLayer(1);
  NonlinearLayer* N2=new NonlinearLayer(15, NonlinearLayer::TANH);
  NonlinearLayer* N3=new NonlinearLayer(1, NonlinearLayer::LINEAR);
  HiddenLayer* H1=new HiddenLayer(15,2);
  HiddenLayer* H2=new HiddenLayer(1,15);
  
  N1->connect_to_last_layer_output(I->layer_output);
  N1->connect_to_next_layer_input(H1->input);
  N2->connect_to_last_layer_output(H1->layer_output);
  N2->connect_to_next_layer_input(H2->input);
  N3->connect_to_last_layer_output(H2->layer_output);
  N3->connect_to_next_layer_input(M->input);
  
  vector<vector<double>> inputdata={{2,3},{3,4},{1,0},{5,5}};
  vector<vector<double>> testdata={{13.2},{25.1},{1},{25}};
  
  double last_loss;
  int i=0;
  while(true){
    i++;
    last_loss=0;
    for(int ii=0;ii<inputdata.size();ii++){
      M->load_data_from_outside(testdata[ii]);
      I->input_data(inputdata[ii]);
      
      I->forward();
      N1->forward();
      H1->forward();
      N2->forward();
      H2->forward();
      N3->forward();
      M->forward();
      
      M->backward();
      N3->backward();
      H2->backward();
      N2->backward();
      H1->backward();
      N1->backward();
      I->backward();
      
      I->train();
      H1->train();
      H2->train();
      if(i%100000==0)cout<<"data: "<<ii<<" predict: "<<H2->layer_output[0]->data<<endl;
      last_loss+=M->layer_output[0]->data;
    }
    if(i%100000==0)cout<<last_loss<<endl;
  }

}
