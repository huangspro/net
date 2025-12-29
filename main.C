#include "Node.h"
#include "Layer.h"
#include "Net.h"
#include<iostream>

using namespace std;

int main(){
  Net N;
  N.add_layer(INPUTLAYER,2);
  N.add_layer(NONLINEARLAYER,2,0,NonlinearLayer::RELU,true);
  N.add_layer(MEANSQUAREERRORLAYER,2);
  N.build();
  vector<double> a={1.25,2};
  vector<double> b={3,4};
  N.train_with_one_data(100,a,b,true);
}
