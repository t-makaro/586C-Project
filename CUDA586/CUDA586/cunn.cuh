#pragma once
#include "cu_utility.cuh"
#include <cassert>
#include <vector>
#include <iostream>
#include <chrono>

typedef std::vector<float> Vector;
typedef std::vector<Vector> Matrix;

class CUNN
{
public:
    CUNN(std::vector<int> layers);
    ~CUNN();
    Vector& forward(const Vector& x, Vector& result);
    void train(const Matrix trainingData, const Vector trainingLabels, const int iterations,
        const int batchSize, float learningRate);
    float evaluate(const Matrix& testData, const std::vector<int>& testLabels);

    void copyWeights(const std::vector<Matrix> weights);
    void copyBiases(const std::vector<Vector> biases);

    static float sigmoid(float x);
    static float d_sigmoid(float x);

protected:
    int numLayers;
    std::vector<int> layers;
    std::vector<Matrix> weights;
    std::vector<Vector> biases;
    // preallocate memory for forward pass
    std::vector<Vector> activations;
    std::vector<Vector> zs;
    // preallocate memory for backward pass
    std::vector<Matrix> dWeights;
    std::vector<Vector> dBiases;

    void updateFromBatch(const Matrix batch, const Vector labels, const float learningRate);

private:

    static Vector& multiply(const Matrix& w, const Vector& x, Vector& result);
    static Vector& add(const Vector& x, const Vector& b, Vector& result);
    Matrix& add(const Matrix& x, const Matrix& b, Matrix& result, const float scale);
    Vector& add(const Vector& x, const Vector& b, Vector& result, const float scale);
    static Vector& sigmoid(Vector& x);
    static Vector& d_sigmoid(Vector& x);
    Vector& sigmoid(const Vector& x, Vector& result);
    static Vector& forwardLayer(const Matrix& w, const Vector& b, const Vector& a,
        Vector& result);

    void backwards(std::vector<Matrix>& dWeights_output, std::vector<Vector>& dBiases_output,
        const Vector& testData, const int testLabel);
    void cost_derivative(const Vector& last_activation, const int label, Vector& result);
    void activation_derivative(const Matrix& weights, Vector& z, Vector& previous);
    Vector& forwardZ(const Matrix& w, const Vector& b, const Vector& a, Vector& result);
    Vector& multiply_elementwise(const Vector& a, const Vector& b, Vector& result);
    Matrix& outer_product(const Vector& a, const Vector& b, Matrix& result);
    void transpose(const Matrix& a, Matrix& result);
};