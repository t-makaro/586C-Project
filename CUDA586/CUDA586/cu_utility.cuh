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

	static std::vector<std::vector<float>>& cuForward(
		const std::vector<float*> d_weights, const std::vector<float*> d_biases,
		const std::vector<float*> d_activations, const std::vector<int> layers,
		const std::vector<std::vector<float>>& X, std::vector<std::vector<float>>& result
	);
	static float* copyDataToDevice(Matrix& X);

	static int* copyDataToDevice(std::vector<int>& X);
	static void testOuterProductAndTranspose(const std::vector<float>& a, const std::vector<float>&b, std::vector<float> &outer, std::vector<float>& transp);
	static void printVector(const std::vector<float>& v, const int& rowLength);

};
