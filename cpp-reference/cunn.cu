#include "cu_utility.cu"
#include "nn.cpp"

class CUNN : public NN {
   private:
    static Vector &forwardLayer(const Matrix &w, const Vector &b,
                                const Vector &a, Vector &result);
    static Vector &multiply(const Matrix &w, const Vector &x, Vector &result);
    static Vector &add(const Vector &x, const Vector &b, Vector &result);
    static Vector &sigmoid(Vector &x);
    static Vector &d_sigmoid(Vector &x);

   public:
    CUNN(NN);
    ~CUNN();
};

Vector &CUNN::forwardLayer(const Matrix &w, const Vector &b, const Vector &a,
                           Vector &result) {
    return result;
}

Vector &CUNN::multiply(const Matrix &w, const Vector &x, Vector &result) {
    cu_utility::cuMatMulVector(w, x, result);
    return result;
}

Vector &CUNN::add(const Vector &x, const Vector &b, Vector &result) {
    return cu_utility::cuVectorAdd(x, b, result);
}

Vector &CUNN::sigmoid(Vector &x) { return cu_utility::cuSigmoid(x); }

Vector &CUNN::d_sigmoid(Vector &x) {
    // TODO: insert return statement here
    return x;
}

CUNN::~CUNN() {}
