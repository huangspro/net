/*
This file is to define the basic calculating node and data node for compute chart
This file also contains the realisation of forward and gradient backward passing
*/

#ifndef _OPE_H_
#define _OPE_H_
#include <cmath>
#include <vector>
#include<iostream>
#include<random>

class Var;
class Ope {
public:
  Ope() {}
  virtual void forward(){}
  virtual void backward() {}
  virtual void load(Var*,Var*,Var*){}
  virtual void load(Var*,Var*){}
  virtual void load_input(Var*){}
  virtual void load_output(Var*){}
  virtual ~Ope() {}
};

class Var {
public:
  double data=0;
  double gradient=0;
  Var(double d, double g) : data(d), gradient(g) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 0.1);
    data=distribution(generator);
  }
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
    a->gradient = out->gradient;
    b->gradient = out->gradient;
  }
};

class SuperAdd : public Ope{
public:
  std::vector<Var*> inputs;
  Var* out;
  SuperAdd(){}
  void load_input(Var* o){
    inputs.push_back(o);
  }
  void load_output(Var* i){
    out=i;
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
  void backward(){
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
    a->gradient = out->gradient * b->data;
    b->gradient = out->gradient * a->data;
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
    a->gradient = out->gradient * (-1.0 / (a->data * a->data));
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
    a->gradient = out->gradient * (-1.0);
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
    a->gradient = out->gradient * (a->data > 0 ? 1.0 : 0.0);
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
    a->gradient = out->gradient * (s * (1.0 - s));
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
    a->gradient = out->gradient * (1.0 - t * t);
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
    a->gradient = out->gradient * out->data;
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
    a->gradient = out->gradient * (1.0 / a->data);
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
    a->gradient = out->gradient * (1.0 / (2.0 * out->data));
  }
};

class Square : public Ope {
public:
  Var *a = nullptr, *out = nullptr;
  Square() {}
  void load(Var* a,Var* b, Var* c){}
  void load(Var* input_a, Var* output) {
    a = input_a; out = output;
  }

  void forward() override {
    out->data = (a->data)*(a->data);
  }
  void backward() override {
    a->gradient = out->gradient * a->data*2;
  }
};
#endif
