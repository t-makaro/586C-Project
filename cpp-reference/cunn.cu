#include "nn.cpp"

class CUNN : public NN
{
private:
    static Vector &forwardLayer(const Matrix &w, const Vector &b, const Vector &a,
                                Vector &result);
    static Vector &multiply(const Matrix &w, const Vector &x, Vector &result);
    static Vector &add(const Vector &x, const Vector &b, Vector &result);
    static Vector &sigmoid(Vector &x);
    static Vector &d_sigmoid(Vector &x);

public:
    CUNN(NN);
    ~CUNN();
};

Vector &CUNN::forwardLayer(const Matrix &w, const Vector &b, const Vector &a, Vector &result)
{
    // TODO: insert return statement here
}

Vector &CUNN::multiply(const Matrix &w, const Vector &x, Vector &result)
{
    assert(result.size() == w.size() && w[0].size() == x.size());

  for (int i = 0; i < w.size(); i++) {
    float sum = 0;
    for (int j = 0; j < w[i].size(); j++) {
      sum += w[i][j] * x[j];
    }
    result[i] = sum;
  }

  return result;
}

Vector &CUNN::add(const Vector &x, const Vector &b, Vector &result)
{
    return cu_utility::cuVectorAdd(x, b, result);
}

Vector &CUNN::sigmoid(Vector &x)
{
    // TODO: insert return statement here
    return cu_utility::cuSigmoid(x)
}

Vector &CUNN::d_sigmoid(Vector &x)
{
    // TODO: insert return statement here
}

CUNN::~CUNN()
{
}
