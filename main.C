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
  
  vector<double> inputdata={15,25};
  vector<double> testdata={-10,-10};
  M->load_data_from_outside(testdata);
  I->input_data(inputdata);
  
  
  I->forward();
  N->forward();
  M->forward();
  M->backward();
  N->backward();
  I->backward();
  cout<<"loss: "<<M->layer_output[0]->data<<endl;
  cout<<"nonoutput: "<<N->layer_output[0]->data<<endl;
  cout<<"nonoutput: "<<N->layer_output[1]->data<<endl;
  cout<<"Minput: "<<M->input[0]->data<<endl;
  cout<<"Minput: "<<M->input[1]->data<<endl;
  cout<<"Minus-output: "<<M->minus_output[0]->data<<endl;
  cout<<"Minus-output: "<<M->minus_output[1]->data<<endl;
  cout<<M->add_output[0]->data<<endl;
  cout<<M->add_output[1]->data<<endl;
}
