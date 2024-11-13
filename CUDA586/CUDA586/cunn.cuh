#pragma once
#include "cu_utility.cuh"
#include <cassert>
#include <vector>
#include <iostream>
#include <chrono>

typedef std::vector<float> Vector;
typedef std::vector<Vector> Matrix;

class CUNN {
public:
    CUNN(std::vector<int> layers);
    ~CUNN();
    Vector& forward(const Vector& x, Vector& result);

    void train(const float* d_trainingData, const int* d_trainingLabels, const int M, const int N,
        const int iterations, const int batchSize, const float learningRate);
    float evaluate(const Matrix& testData, const std::vector<int>& testLabels);
    float evaluate(const float *input, const int* labels, int numExamples);

    void copyWeights(const std::vector<Matrix> weights);
    void copyBiases(const std::vector<Vector> biases);

    static float sigmoid(float x);
    static float d_sigmoid(float x);

    void copyParametersToDevice();
    void testForwardZ(bool isGpu, Vector &testData);
    void testBackwardOutputLayer(bool isGPU, Vector& testData, int testLabel);
    void setBatchSizeDevice(int batchSize);

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

    void updateFromBatch(const float* d_batch, const int* d_labels,
        const int batchSize, const int dataLen, const float learningRate);

    // device pointers
    std::vector<float *> d_weights;
    std::vector<float *> d_biases;
    std::vector<float *> d_activations;
    std::vector<float*> d_zs;

	int batchSize = 1;
;   std::vector<float *> d_weights_batch;
    std::vector<float *> d_biases_batch;
    std::vector<float *> d_activations_batch;
    
;
    void deviceAlloc();

	void deviceFree();


private:
    static Vector& multiply(const Matrix& w, const Vector& x, Vector& result);
    static Vector& add(const Vector& x, const Vector& b, Vector& result);
    Matrix& add(const Matrix& x, const Matrix& b, Matrix& result,
        const float scale);
    Vector& add(const Vector& x, const Vector& b, Vector& result,
        const float scale);
    static Vector& sigmoid(Vector& x);
    static Vector& d_sigmoid(Vector& x);
    Vector& d_sigmoid(const Vector& x, Vector& y);
    Vector& sigmoid(const Vector& x, Vector& result);
    static Vector& forwardLayer(const Matrix& w, const Vector& b,
        const Vector& a, Vector& result);

    void backwards(std::vector<float*> &dWeights_output,
        std::vector<float*> &dBiases_output,
        const float* testData, const int* testLabel, size_t dataLen);
    void cost_derivative(const Vector& last_activation, const int label,
        Vector& result);
    void activation_derivative(const Matrix& weights, Vector& z,
        Vector& previous);
    Vector& forwardZ(const Matrix& w, const Vector& b, const Vector& a,
        Vector& result);
    Vector& multiply_elementwise(const Vector& a, const Vector& b,
        Vector& result);
    Matrix& outer_product(const Vector& a, const Vector& b, Matrix& result);
    void transpose(const Matrix& a, Matrix& result);

    std::vector<float*> allocate_like_weights();
    std::vector<float*> allocate_like_biases();
};