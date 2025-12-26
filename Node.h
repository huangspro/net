#ifndef _NODE_H_
#define _NODE_H_
#include <cmath>
#include <vector>

class Var {
public:
    double data;
    double gradient;
    Var(double d, double g = 0.0) : data(d), gradient(g) {}
};

class Node {
public:
    Node() {}
    virtual double forword() = 0; // 统一拼写
    virtual void backward() = 0;
    virtual ~Node() {}
};

// =====================================================================================

class Add : public Node {
public:
    Var *a, *b, *out;
    Add(Var* input_a, Var* input_b, Var* output) : a(input_a), b(input_b), out(output) {}
    
    double forword() override {
        out->data = a->data + b->data;
        return out->data;
    }
    void backward() override {
        a->gradient += out->gradient;
        b->gradient += out->gradient;
    }
};

class Mul : public Node {
public:
    Var *a, *b, *out;
    Mul(Var* input_a, Var* input_b, Var* output) : a(input_a), b(input_b), out(output) {}

    double forword() override {
        out->data = a->data * b->data;
        return out->data;
    }
    void backward() override {
        a->gradient += out->gradient * b->data;
        b->gradient += out->gradient * a->data;
    }
};

class Dev : public Node {
public:
    Var *a, *out;
    Dev(Var* input_a, Var* output) : a(input_a), out(output) {}

    double forword() override {
        out->data = 1.0 / a->data;
        return out->data;
    }
    void backward() override {
        a->gradient += out->gradient * (-1.0 / (a->data * a->data));
    }
};

class Minus : public Node {
public:
    Var *a, *out;
    Minus(Var* input_a, Var* output) : a(input_a), out(output) {}

    double forword() override {
        out->data = -1.0 * a->data;
        return out->data;
    }
    void backward() override {
        a->gradient += out->gradient * (-1.0);
    }
};

class Relu : public Node {
public:
    Var *a, *out;
    Relu(Var* input_a, Var* output) : a(input_a), out(output) {}

    double forword() override {
        out->data = a->data > 0 ? a->data : 0;
        return out->data;
    }
    void backward() override {
        a->gradient += out->gradient * (a->data > 0 ? 1.0 : 0.0);
    }
};

class Sigmoid : public Node {
public:
    Var *a, *out;
    Sigmoid(Var* input_a, Var* output) : a(input_a), out(output) {}

    double forword() override {
        out->data = 1.0 / (1.0 + exp(-a->data));
        return out->data;
    }
    void backward() override {
        double s = out->data; // 直接利用forward存好的结果
        a->gradient += out->gradient * (s * (1.0 - s));
    }
};

class Tanh : public Node {
public:
    Var *a, *out;
    Tanh(Var* input_a, Var* output) : a(input_a), out(output) {}

    double forword() override {
        out->data = tanh(a->data);
        return out->data;
    }
    void backward() override {
        double t = out->data;
        a->gradient += out->gradient * (1.0 - t * t);
    }
};

class Exp : public Node {
public:
    Var *a, *out;
    Exp(Var* input_a, Var* output) : a(input_a), out(output) {}

    double forword() override {
        out->data = exp(a->data);
        return out->data;
    }
    void backward() override {
        a->gradient += out->gradient * out->data;
    }
};

class Ln : public Node {
public:
    Var *a, *out;
    Ln(Var* input_a, Var* output) : a(input_a), out(output) {}

    double forword() override {
        out->data = log(a->data);
        return out->data;
    }
    void backward() override {
        a->gradient += out->gradient * (1.0 / a->data);
    }
};

class Pow : public Node {
public:
    Var *a, *out;
    double exponent;
    Pow(Var* input_a, Var* output, double e) : a(input_a), out(output), exponent(e) {}

    double forword() override {
        out->data = pow(a->data, exponent);
        return out->data;
    }
    void backward() override {
        a->gradient += out->gradient * (exponent * pow(a->data, exponent - 1));
    }
};

class Sqrt : public Node {
public:
    Var *a, *out;
    Sqrt(Var* input_a, Var* output) : a(input_a), out(output) {}

    double forword() override {
        out->data = sqrt(a->data);
        return out->data;
    }
    void backward() override {
        a->gradient += out->gradient * (1.0 / (2.0 * out->data));
    }
};

class SquareDiff : public Node {
public:
    Var *a, *b, *out;
    SquareDiff(Var* input_a, Var* input_b, Var* output) : a(input_a), b(input_b), out(output) {}

    double forword() override {
        double diff = a->data - b->data;
        out->data = diff * diff;
        return out->data;
    }
    void backward() override {
        double diff = a->data - b->data;
        a->gradient += out->gradient * (2.0 * diff);
        b->gradient += out->gradient * (-2.0 * diff);
    }
};

#endif
