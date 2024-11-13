#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>

typedef std::vector<float> Vector;
typedef std::vector<Vector> Matrix;

// APIs
class cu_utility {
private:
	/* data */
public:
	cu_utility(/* args */);
	~cu_utility();
	static void d_VectorAdd(float* A, float* B, float* result, float N, float scale);
	static std::vector<float>& cuVectorAdd(const std::vector<float>& x,
		const std::vector<float>& b,
		std::vector<float>& result);
	static std::vector<float>& cuSigmoid(std::vector<float>& x);
	static std::vector<float>& cuDSigmoid(std::vector<float>& x);
	static std::vector<float>& cuMatMulVector(
		const std::vector<std::vector<float>>& W, const std::vector<float>& x,
		std::vector<float>& result);
	static std::vector<float>& cuForwardLayer(
		const std::vector<std::vector<float>>& W, const std::vector<float>& b,
		const std::vector<float>& x, std::vector<float>& result);
	static std::vector<float>& cuForwardLayerWithZs(
		const std::vector<std::vector<float>>& W, const std::vector<float>& b,
		const std::vector<float>& x, std::vector<float>& zsi,std::vector<float>& result);
	static void cuForwardLayerWithZs(const float* d_W, const float* d_b, const float* d_x, float* d_zsi, float* d_y, int M, int N);
	static void cuBackwardOutputLayer(float* d_outActivation, float* d_inActivation,
		float* d_bias_output, float* d_weight_output,
		float* d_zsi, float* d_delta, const int* d_testLabel, int inSize, int outSize);
	static void cuBackwardRegularLayer(float* d_inActivation, float* d_bias_output, float* d_weight_input,
	                            float* d_dWeight_output,
	                            float* d_zsi_in, float* d_zsi_out, float* d_delta_in, float* d_delta_out, int inSize,
	                            int outSize, int deltaSize);


	static std::vector<std::vector<float>>& cuForward(
		const std::vector<float*> d_weights, const std::vector<float*> d_biases,
		const std::vector<float*> d_activations, const std::vector<int> layers,
		const std::vector<std::vector<float>>& X, std::vector<std::vector<float>>& result
	);
	static int cuForwardBatch(
		const std::vector<float*> d_weights, const std::vector<float*> d_biases,
		const std::vector<float*> d_activations_batch, const std::vector<int> layers,
		const float *d_X, 
		const int* d_Y,
		int numExamples,
		int batchSize
	);
	static float* copyDataToDevice(Matrix& X);
	static int* copyDataToDevice(std::vector<int>& X);

	static void testOuterProductAndTranspose(const std::vector<float>& a, const std::vector<float>&b, std::vector<float> &outer, std::vector<float>& transp);
	static void printVector(const std::vector<float>& v, const int& rowLength);

};
