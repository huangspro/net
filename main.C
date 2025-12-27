#include "Node.h"
#include "net.h"
#include<iostream>

using namespace std;

int main(){
  InputLayer* I = new InputLayer(10);
  I->connect_to_nonlinear_layer(new NonlinearLayer(10,NonlinearLayer::RELU));
}
