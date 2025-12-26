#ifndef _CHART_H_
#define _CHART_H_

#include "Node.h"
#include<vector>

class Chart{
public:
  std::vector<Var*> allvars;
  std::vector<Ope*> allopes;
  void forward(){
    for(auto i=allopes.begin();i!=allopes.end();i++){
      (*i)->forward();
    }
  }
  void backward(){
    for(auto i=allopes.rbegin();i!=allopes.rend();i++){
      (*i)->backward();
    }
  }
  
};
#endif
