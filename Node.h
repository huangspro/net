/*
This file is to define the basic calculating node and data node for compute chart
This file also contains the realisation of forward and gradient backward passing
*/

#ifndef _OPE_H_
#define _OPE_H_
#include <cmath>
#include <vector>

class Var;
class Ope {
public:
  Ope() {}
  virtual void forward(){}
  virtual void backward() {}
  virtual void load(Var*,Var*,Var*){}
  virtual void load(Var*,Var*){}
  virtual void load(Var*){}
  virtual ~Ope() {}
};

class Var {
public:
  double data;
  double gradient;
  Var(double d, double g) : data(d), gradient(g) {}
};

// =====================================================================================

class Add : public Ope {
public:
  Var *a = nullptr, *b = nullptr, *out = nullptr;
  Add() {}
  void load(Var* a,Var* b){}
  void load(Var* input_a, Var* input_b, Var* output) {
    a = input_a; b = input_b; out = output;
  }

  void forward() override {
    out->data = a->data + b->data;
  }
  void backward() override {
    a->gradient += out->gradient;
    b->gradient += out->gradient;
  }
};

class SuperAdd : public Ope{
public:
  std::vector<Var*> inputs;
  Var* out;
  SuperAdd(){}
  void load(Var* o){
    inputs.push_back(o);
  }
  void forward(){
    double tem=0;
    for(auto i=inputs.begin();i!=inputs.end();i++){
      tem+=(*i)->data;
    }
    out->data=tem;
  }
  void load(Var* a, Var* b, Var* c){}
  void load(Var* a, Var* b){}
  void backfard(){
    for(auto i=inputs.begin();i!=inputs.end();i++){
      (*i)->gradient=out->gradient;
    }
  }
};

class Mul : public Ope {
public:
  Var *a = nullptr, *b = nullptr, *out = nullptr;
  Mul() {}
  void load(Var* a,Var* b){}
  void load(Var* input_a, Var* input_b, Var* output) {
    a = input_a; b = input_b; out = output;
  }

  void forward() override {
    out->data = a->data * b->data;
    
  }
  void backward() override {
    a->gradient += out->gradient * b->data;
    b->gradient += out->gradient * a->data;
  }
};

class Dev : public Ope {
public:
  Var *a = nullptr, *out = nullptr;
  Dev() {}
  void load(Var* a,Var* b, Var* c){}  
  void load(Var* input_a, Var* output) {
    a = input_a; out = output;
  }

  void forward() override {
    out->data = 1.0 / a->data;
    
  }
  void backward() override {
    a->gradient += out->gradient * (-1.0 / (a->data * a->data));
  }
};

class Minus : public Ope {
public:
  Var *a = nullptr, *out = nullptr;
  Minus() {}
  void load(Var* a,Var* b, Var* c){}
  void load(Var* input_a, Var* output) {
    a = input_a; out = output;
  }

  void forward() override {
    out->data = -1.0 * a->data;
    
  }
  void backward() override {
    a->gradient += out->gradient * (-1.0);
  }
};

class Relu : public Ope {
public:
  Var *a = nullptr, *out = nullptr;
  Relu() {}
  void load(Var* a,Var* b, Var* c){}
  void load(Var* input_a, Var* output) {
    a = input_a; out = output;
  }

  void forward() override {
    out->data = a->data > 0 ? a->data : 0;
    
  }
  void backward() override {
    a->gradient += out->gradient * (a->data > 0 ? 1.0 : 0.0);
  }
};

class Sigmoid : public Ope {
public:
  Var *a = nullptr, *out = nullptr;
  Sigmoid() {}
  void load(Var* a,Var* b, Var* c){}
  void load(Var* input_a, Var* output) {
    a = input_a; out = output;
  }

  void forward() override {
    out->data = 1.0 / (1.0 + exp(-a->data));
    
  }
  void backward() override {
    double s = out->data; 
    a->gradient += out->gradient * (s * (1.0 - s));
  }
};

class Tanh : public Ope {
public:
  Var *a = nullptr, *out = nullptr;
  Tanh() {}
  void load(Var* a,Var* b, Var* c){}
  void load(Var* input_a, Var* output) {
    a = input_a; out = output;
  }

  void forward() override {
    out->data = tanh(a->data);
    
  }
  void backward() override {
    double t = out->data;
    a->gradient += out->gradient * (1.0 - t * t);
  }
};

class Exp : public Ope {
public:
  Var *a = nullptr, *out = nullptr;
  Exp() {}
  void load(Var* a,Var* b, Var* c){}
  void load(Var* input_a, Var* output) {
    a = input_a; out = output;
  }

  void forward() override {
    out->data = exp(a->data);
    
  }
  void backward() override {
    a->gradient += out->gradient * out->data;
  }
};

class Ln : public Ope {
public:
  Var *a = nullptr, *out = nullptr;
  Ln() {}
  void load(Var* a,Var* b, Var* c){}
  void load(Var* input_a, Var* output) {
    a = input_a; out = output;
  }

  void forward() override {
    out->data = log(a->data);
    
  }
  void backward() override {
    a->gradient += out->gradient * (1.0 / a->data);
  }
};



class Sqrt : public Ope {
public:
  Var *a = nullptr, *out = nullptr;
  Sqrt() {}
  void load(Var* a,Var* b, Var* c){}
  void load(Var* input_a, Var* output) {
    a = input_a; out = output;
  }

  void forward() override {
    out->data = sqrt(a->data);
    
  }
  void backward() override {
    a->gradient += out->gradient * (1.0 / (2.0 * out->data));
  }
};

class SquareDiff : public Ope {
public:
  Var *a = nullptr, *b = nullptr, *out = nullptr;
  SquareDiff() {}
  void load(Var* a,Var* b){}
  void load(Var* input_a, Var* input_b, Var* output) {
    a = input_a; b = input_b; out = output;
  }

  void forward() override {
    double diff = a->data - b->data;
    out->data = diff * diff;
    
  }
  void backward() override {
    double diff = a->data - b->data;
    a->gradient += out->gradient * (2.0 * diff);
    b->gradient += out->gradient * (-2.0 * diff);
  }
};

#endif
