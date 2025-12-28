#include "Node.h"
#include "Layer.h"
#include<iostream>

using namespace std;

int main(){
  InputLayer* I = new InputLayer(2);
  NonlinearLayer* N1 = new NonlinearLayer(2,NonlinearLayer::RELU);
  HiddenLayer* H = new HiddenLayer(2,10);
  NonlinearLayer* N2 = new NonlinearLayer(2,NonlinearLayer::RELU);
  
  N1->connect_to_last_layer_output(I->layer_output);
  N1->connect_to_next_layer_input(H->input);
  N2->connect_to_last_layer_output(H->layer_output);
}
