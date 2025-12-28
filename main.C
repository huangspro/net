#include "Node.h"
#include "Layer.h"
#include<iostream>

using namespace std;

int main(){
  InputLayer* I = new InputLayer(10);
  for(int i=0;i<10;i++){
    I->input[i]->data=1.5;
  }
  NonlinearLayer* N1 = new NonlinearLayer(10,NonlinearLayer::RELU);
  HiddenLayer* H = new HiddenLayer(20,10);
  NonlinearLayer* N2 = new NonlinearLayer(20,NonlinearLayer::RELU);
  SoftmaxLayer* S = new SoftmaxLayer(20);
  N1->connect_to_last_layer_output(I->layer_output);
  N1->connect_to_next_layer_input(H->input);
  N2->connect_to_last_layer_output(H->layer_output);
  N2->connect_to_next_layer_input(S->input);
  I->forward();
  N1->forward();
  H->forward();
  N2->forward();
  S->forward();
  for(int i=0;i<20;i++){
    cout<<S->layer_output[i]->data<<endl;
    S->layer_output[i]->gradient=1;
  }
  S->backward();
  N2->backward();
  H->backward();
  N1->backward();
  I->backward();
  for(int i=0;i<10;i++){
    cout<<I->weight[i]->gradient<<endl;
  }
}
