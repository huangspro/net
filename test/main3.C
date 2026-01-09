#include "../base_class/Layer.h"
#include "../base_class/ConvolutionLayer.h"
#include<iostream>

using namespace std;

int main(){
  // 创建层
  ConvolutionLayer* CON = new ConvolutionLayer(2, 2, 3, 3, 1);  
  NonlinearLayer* N0 = new NonlinearLayer(4, NonlinearLayer::RELU);  
  InputLayer* I = new InputLayer(4);                            
  NonlinearLayer* N1 = new NonlinearLayer(4, NonlinearLayer::RELU);  
  HiddenLayer* H1 = new HiddenLayer(30, 4);                   
  NonlinearLayer* N2 = new NonlinearLayer(30, NonlinearLayer::RELU); 
  HiddenLayer* H2 = new HiddenLayer(20, 30);                    
  NonlinearLayer* N3 = new NonlinearLayer(20, NonlinearLayer::RELU); 
  HiddenLayer* H3 = new HiddenLayer(3, 20);                     
  NonlinearLayer* N4 = new NonlinearLayer(3, NonlinearLayer::SIGMOID);  
  MeanSquareErrorLayer* C = new MeanSquareErrorLayer(3);         

// 连接层
  CON->connect_to_next_layer_input(N0->input);
  
  
  N0->connect_to_next_layer_input(I->input);

  N1->connect_to_last_layer_output(I->layer_output);
  N1->connect_to_next_layer_input(H1->input);
  
  N2->connect_to_last_layer_output(H1->layer_output);
  N2->connect_to_next_layer_input(H2->input);

  N3->connect_to_last_layer_output(H2->layer_output);
  N3->connect_to_next_layer_input(H3->input);

  N4->connect_to_last_layer_output(H3->layer_output);
  N4->connect_to_next_layer_input(C->input);
  
  vector<vector<vector<double>>> inputdata = {
    // L 字母示例
    {
        {1,0,0},
        {1,0,0},
        {1,1,0}
    },
    {
        {1,0,0},
        {1,0,0},
        {1,1,0}
    },
    {
        {0,1,0},
        {0,1,0},
        {0,1,1}
    },
    {
        {0,0,0},
        {0,0,1},
        {1,1,1}
    },
    {
        {0,0,1},
        {0,0,1},
        {1,1,1}
    },

    // 锐角折线示例
    {
        {0,0,0},
        {0,1,0},
        {1,1,0}
    },
    {
        {1,1,0},
        {0,1,0},
        {0,0,0}
    },
    {
        {0,1,0},
        {1,1,0},
        {0,0,0}
    },
    {
        {0,0,0},
        {0,1,0},
        {1,1,0}
    },
    {
        {0,0,0},
        {1,1,0},
        {1,0,0}
    },

    // 小方形 2x2
    {
        {1,1,0},
        {1,1,0},
        {0,0,0}
    },
    {
        {0,1,1},
        {0,1,1},
        {0,0,0}
    },
    {
        {1,1,0},
        {1,1,0},
        {0,0,0}
    },
    {
        {0,0,0},
        {1,1,0},
        {1,1,0}
    },
    {
        {0,0,0},
        {0,1,1},
        {0,1,1}
    }
};/*
    vector<vector<double>> inputdata = {
    // L 字母示例
    {1,0,0, 1,0,0, 1,1,0},
    {1,0,0, 1,0,0, 1,1,0},
    {0,1,0, 0,1,0, 0,1,1},
    {0,0,0, 0,0,1, 1,1,1},
    {0,0,1, 0,0,1, 1,1,1},

    // 锐角折线示例
    {0,0,0, 0,1,0, 1,1,0},
    {1,1,0, 0,1,0, 0,0,0},
    {0,1,0, 1,1,0, 0,0,0},
    {0,0,0, 0,1,0, 1,1,0},
    {0,0,0, 1,1,0, 1,0,0},

    // 小方形 2x2
    {1,1,0, 1,1,0, 0,0,0},
    {0,1,1, 0,1,1, 0,0,0},
    {1,1,0, 1,1,0, 0,0,0},
    {0,0,0, 1,1,0, 1,1,0},
    {0,0,0, 0,1,1, 0,1,1}
};*/
  vector<vector<double>> testdata = {
    {1,0,0},{1,0,0},{1,0,0},{1,0,0},{1,0,0},
    {0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,1,0},
    {0,0,1},{0,0,1},{0,0,1},{0,0,1},{0,0,1}
  };
  
  double last_loss;
  int i=0;
  while(i<100000){
    i++;
    last_loss=0;
    for(int ii=0;ii<inputdata.size();ii++){
      C->load_data_from_outside(testdata[ii]);
      CON->load_data_from_outside(inputdata[ii]);
      
      CON->forward();
      N0->forward();
      I->forward();
      N1->forward();
      H1->forward();
      N2->forward();
      H2->forward();
      N3->forward();
      H3->forward();
      N4->forward();
      C->forward();
      
      C->backward();
      N4->backward();
      H3->backward();
      N3->backward();
      H2->backward();
      N2->backward();
      H1->backward();
      N1->backward();
      I->backward();
      N0->backward();
      CON->backward();
      
      I->train();
      H1->train();
      H2->train();
      H3->train();
      
      
      if(i%10000==0)cout<<CON->conkernel[0][0]->gradient<<" "<<CON->conkernel[0][1]->gradient<<endl;
      if(i%10000==0)cout<<CON->conkernel[1][0]->gradient<<" "<<CON->conkernel[1][1]->gradient<<endl;
      if(i%10000==0)cout<<N3->input[0]->gradient<<endl;
      if(i%10000==0)cout<<"真实值: "<<testdata[ii][0]<<testdata[ii][1]<<testdata[ii][2]<<" 预测: "<<(N4->layer_output[0]->data>0.5?'1':'0')<<(N4->layer_output[1]->data>0.5?'1':'0')<<(N4->layer_output[2]->data>0.5?'1':'0')<<endl;
      last_loss+=C->layer_output[0]->data;
    }
    CON->train();
    if(i%10000==0)cout<<last_loss/15<<endl;
  }
  //cout<<CON->conkernel[0][0]->data<<" "<<CON->conkernel[0][1]->data<<endl;
  //cout<<CON->conkernel[1][0]->data<<" "<<CON->conkernel[1][1]->data<<endl;
  /*delete C; 
  delete CON;
  delete N4;
  delete H3;
  delete N3;
  delete H2;  gradient
  delete N2;
  delete H1;
  delete N1;
  delete I; */  
}
