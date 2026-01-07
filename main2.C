#include "base_class/Layer.h"
#include<iostream>

using namespace std;

int main(){
  InputLayer* I=new InputLayer(3);
  NonlinearLayer* N1=new NonlinearLayer(3, NonlinearLayer::TANH);
  HiddenLayer* H1=new HiddenLayer(10,3);
  NonlinearLayer* N2=new NonlinearLayer(10, NonlinearLayer::TANH);
  HiddenLayer* H2=new HiddenLayer(1,10);
  NonlinearLayer* N3=new NonlinearLayer(1, NonlinearLayer::SIGMOID);
  MeanSquareErrorLayer* C=new MeanSquareErrorLayer(1);
  
  N1->connect_to_last_layer_output(I->layer_output);
  N1->connect_to_next_layer_input(H1->input);
  N2->connect_to_last_layer_output(H1->layer_output);
  N2->connect_to_next_layer_input(H2->input);
  N3->connect_to_last_layer_output(H2->layer_output);
  N3->connect_to_next_layer_input(C->input);
  
  vector<vector<double>> inputdata = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {0, 255, 255},
        {255, 165, 0}, {128, 0, 128}, {0, 128, 128}, {192, 192, 192}, {255, 192, 203},
        {139, 0, 0}, {0, 100, 0}, {0, 0, 139}, {184, 134, 11}, {0, 139, 139},
        {255, 69, 0}, {147, 112, 219}, {0, 206, 209}, {105, 105, 105}, {255, 105, 180},
        {178, 34, 34}, {34, 139, 34}, {25, 25, 112}, {218, 165, 32}, {64, 224, 208},
        {255, 140, 0}, {186, 85, 211}, {72, 209, 204}, {169, 169, 169}, {255, 20, 147},
        {220, 20, 60}, {50, 205, 50}, {70, 130, 180}, {238, 232, 170}, {175, 238, 238},
        {255, 69, 0}, {148, 0, 211}, {0, 191, 255}, {112, 128, 144}, {255, 182, 193},
        {178, 34, 34}, {0, 128, 0}, {65, 105, 225}, {250, 250, 210}, {0, 255, 127},
        {255, 99, 71}, {147, 112, 219}, {0, 206, 209}, {211, 211, 211}, {255, 182, 193}
    };

    // 输出数据: 50 条冷暖标签 (1=暖色, 0=冷色)
    vector<vector<double>> testdata = {
        {1}, {0}, {0}, {1}, {0},
        {1}, {1}, {0}, {0}, {1},
        {1}, {0}, {0}, {1}, {0},
        {1}, {1}, {0}, {0}, {1},
        {1}, {0}, {0}, {1}, {0},
        {1}, {1}, {0}, {0}, {1},
        {1}, {0}, {0}, {1}, {0},
        {1}, {1}, {0}, {0}, {1},
        {1}, {0}, {0}, {1}, {0},
        {1}, {1}, {0}, {0}, {1}
    };
  
  double last_loss;
  int i=0;
  while(true){
    i++;
    last_loss=0;
    for(int ii=0;ii<inputdata.size();ii++){
      C->load_data_from_outside(testdata[ii]);
      I->input_data(inputdata[ii]);
      
      I->forward();
      N1->forward();
      H1->forward();
      N2->forward();
      H2->forward();
      N3->forward();
      C->forward();
      
      C->backward();
      N3->backward();
      H2->backward();
      N2->backward();
      H1->backward();
      N1->backward();
      I->backward();
      
      I->train();
      H1->train();
      H2->train();
      if(i%100000==0)cout<<"真实值: "<<testdata[ii][0]<<" 预测: "<<(N3->layer_output[0]->data>0.5?'1':'0')<<endl;
      last_loss+=C->layer_output[0]->data;
    }
    if(i%100000==0){cout<<last_loss/50<<endl;}
  }

  delete C;   
  delete N3;
  delete H2;  
  delete N2;
  delete H1;
  delete N1;
  delete I;   
}
