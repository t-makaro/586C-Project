#include "cu_utility.cuh"

#include "utility.h"
// Device Kernels

__device__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// in-place
// W of shape MxN, b of shape 1xN
__device__ void matAddVec(float* WX, const float* b, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
		WX[i * N + j] += b[j];
    }
}

__device__ void matMulVec(const float* W, const float* X, float* Y, int M,
    int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float tmp = 0.0;
        for (int k = 0; k < N; k++) {
            tmp += W[i * N + k] * X[k];
        }
        Y[i] = tmp;
    }
}

__device__ void matMul(const float* W, const float* X, float* result, int M, int N, int K) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < K) {
        float tmp = 0.0;
        for (int k = 0; k < N; k++) {
			tmp += W[i * N + k] * X[k * K + j];
        }
        result[i * K + j] = tmp;
    }
}


__device__ float sigmoid(float a) { return 1.0 / (1.0 + exp(-a)); }

__device__ void sigmoid(float* A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        A[i] = sigmoid(A[i]);
    }
}

__device__ void sigmoidMat(float* A, int M, int N) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < M && j < N) {
		A[i * N + j] = sigmoid(A[i * N + j]);
	}
}

__device__ float d_sigmoid(float a) {
    float xp = exp(-a);
    return xp / ((1.0 + xp) * (1.0 + xp));
}

__device__ void d_sigmoid(float* A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        A[i] = d_sigmoid(A[i]);
    }
}

__device__ void d_sigmoid_non_inplace(float* A, int N, float* Res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Res[i] = d_sigmoid(A[i]);
    }
}

__device__ void sigmoid_non_inplace(float* A, int N, float* Res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Res[i] = sigmoid(A[i]);
    }
}


__device__ void outer_product(const float* A, const float* B, int M, int N, float* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) { //M*N = 784*300 = 235,200. That seems like a bad idea.
        int row = i / N;
        int col = i % N;
        result[i] = A[row] * B[col];
    }
}

__device__ void transpose(const float* input, float* output, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M * N) {
        int row = i / N; // Original Alignment: 
        int col = i % N;

        output[col * M + row] = input[i];
    }
}

__device__ void transposeMultiply(const float* mat, const float* vec, float* output, int M, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float sum = 0.0f;
        for (int col = 0; col < M; ++col) {
            sum += mat[col * N + row] * vec[col];
        }
        output[row] = sum;
    }
}

__device__ void multiply_elementwise(const float* A, const float* B, int N, float* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        result[i] = A[i] * B[i];
    }
}

__device__ void cost_derivative(const float* last_activation, const int label, const int outLayerLength, float* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < outLayerLength) // outLayerLength is always 10 (for the 10 digits)
    {
        if (i == label) {
            result[i] = -1.0f / (last_activation[i] + FLT_EPSILON);
            //result[i] = __fdividef(-1.0f, last_activation[i] + FLT_EPSILON);
        }
        else {
            result[i] = 1.0f / (1.0f - last_activation[i] + FLT_EPSILON);
            //result[i] = __fdividef(-1.0f, 1.0f - last_activation[i] + FLT_EPSILON);
        }
    }
}


__device__ void activation_derivative(const float* d_weight, float* d_zsi, float* d_zstemp, float* d_prev_delta, float* d_new_delta, int inLayerLength, int outLayerLength)
{
	// TODO
    // Safe, in-place operation
    d_sigmoid_non_inplace(d_zsi, outLayerLength, d_zstemp);
    multiply_elementwise(d_zstemp, d_prev_delta, outLayerLength, d_prev_delta);
    // Might need to use the global kernel to sync...
    transposeMultiply(d_weight, d_prev_delta, d_new_delta, outLayerLength, inLayerLength);
}

__global__ void global_test_kernel_matTran_outerProduct(const float* A, const float* B, int M, int N, float* resOuterProduct, float* resTranspose)
{
    outer_product(A, B, M, N, resOuterProduct);
    transpose(resOuterProduct, resTranspose, M, N);
}


// Global Kernels
__global__ void global_vectorAdd(const float* A, const float* B, float* C,
    int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
__global__ void global_vectorAdd(const float* A, const float* B, float* C,
    int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + scale*B[i];
    }
}

__global__ void global_matMulVec(const float* W, const float* X, float* Y,
    int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float tmp = 0.0;
        for (int k = 0; k < N; k++) {
            tmp += W[i * N + k] * X[k];
        }
        Y[i] = tmp;
    }
}

__global__ void global_sigmoid(float* A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        A[i] = sigmoid(A[i]);
    }
}

__global__ void global_d_sigmoid(float* A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        A[i] = d_sigmoid(A[i]);
    }
}

__global__ void global_forwardLayer(const float* W, const float* b,
    const float* A, float* result, int M,
    int N) {
    // multiply
    matMulVec(W, A, result, M, N);
    // add
    vectorAdd(result, b, result, M);
    // activate
    sigmoid(result, M);
}

__global__ void global_forwardLayer_zsi(const float* W, const float* b, const float* A, float* result, float* zsi, int M, int N)
{
    // We need the intermediate value Zs in the backward pass.
    matMulVec(W, A, zsi, M, N);
    vectorAdd(zsi, b, zsi, M);
    sigmoid_non_inplace(zsi, M, result);
}

__global__ void global_backwardLayer_output(const float* outActivation, float* bias_output, float* zsi, float* zstemp, float* delta,const int* d_testLabel, int outLayerLength)
{
    // Output layer of backward propagation. Running Cost Derivative
    // For our problem, this outLayerLength can be hardcoded as 10
    cost_derivative(outActivation, *d_testLabel, outLayerLength, delta); // Keep in mind we are moving backwards. outActivation is the output layer and inActivation is the input layer.
    d_sigmoid_non_inplace(zsi, outLayerLength, zstemp);
    multiply_elementwise(zstemp, delta, outLayerLength, bias_output);
    // Outer product has to be called from another global kernel. It will need all blocks to finish calculating bias_output

    // __syncthreads() // will not work here as the number of threads we are launching here is more than the capacity of a block! 
    // outer_product(bias_output, inActivation, outLayerLength, inLayerLength, weight_output);

    // Do the first 3 device kernels need this treatment? No.
    // zsi is not changed. delta is changed but it's only changed in place by the same thread (i.e. delta[i] will only be changed by the same thread that operated zstemp[i], so we can ensure zstemp's value is safe
    // but it's not safe for outer_product! That kernel uses all the bias_outputs!
}

__global__ void global_outer_product(const float* bias_output, const float* inActivation, float* weight_output, int outLayerLength, int inLayerLength)
{
    outer_product(bias_output, inActivation, outLayerLength, inLayerLength, weight_output);
}

__global__ void global_d_sigmoid_multiply_elementwise_delta(float* d_zsi, float* d_new, float* d_delta, float* d_zstemp, int outLayerLength)
{
	// Perform a d_sigmoid_non_inplace then multiply elementwise.
    d_sigmoid_non_inplace(d_zsi, outLayerLength, d_zstemp);
    multiply_elementwise(d_zstemp, d_delta, outLayerLength, d_new);
}

__global__ void global_matTranMul(const float* mat, const float* vec, int M, int N, float* res)
{
    transposeMultiply(mat, vec, res, M, N);
}

__device__ void batched_forwardMatMul(const float* W, const float* A, float* result, int M, int N, int batchSize, int i, int j)
{
	float tmp = 0.0;
	for (int k = 0; k < N; k++) {
		tmp += W[i * N + k] * A[j * N + k];
	}
	result[j * M + i] = tmp;
}
__global__ void batched_forwardMatMul(const float* W, const float* A, float* result, int M, int N, int batchSize)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < M && j < batchSize) {
		batched_forwardMatMul(W, A, result, M, N, batchSize, i, j);
	}
}

__device__ void batched_addBiases(const float* b, const float* inputMatrix, float* result, int M, int batchSize, int i, int j)
{
	result[j * M + i] = inputMatrix[j * M + i] + b[i];
}
__global__ void batched_addBiases(const float* b, const float* inputMatrix, float* result, int M, int batchSize)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < M && j < batchSize) {
		batched_addBiases(inputMatrix, b, result, M, batchSize, i, j);
	}
}

__device__ void batched_sigmoid(const float* Z, float* result, int M, int batchSize, int i, int j)
{
	result[j * M + i] = sigmoid(Z[j * M + i]);
}
__global__ void batched_sigmoid(const float* Z, float* result, int M, int batchSize)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < M && j < batchSize) {
		batched_sigmoid(Z, result, M, batchSize, i, j);
	}
}

__global__ void global_forwardLayerBatch(const float* W, const float* b, const float* A, float* result, int M, int N, int batchSize)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < batchSize) {
		batched_forwardMatMul(W, A, result, M, N, batchSize, i, j);
		batched_addBiases(b, result, result, M, batchSize, i, j);
        batched_sigmoid(result, result, M, batchSize, i, j);
    }
}

__global__ void countNumCorrectPredictions(const float* predictions, const int* labels, int* numCorrect, int numExamples, int numClasses) {
    int i= blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numExamples) {
		int maxIndex = 0;
		float maxVal = predictions[i * numClasses];
		for (int j = 1; j < numClasses; j++) {
			float val = predictions[i * numClasses + j];
			if (val > maxVal) {
				maxVal = val;
				maxIndex = j;
			}
		}
		atomicAdd(numCorrect, (maxIndex == labels[i]));
    }
}


cu_utility::cu_utility(/* args */) {}

cu_utility::~cu_utility() {}

void cu_utility::d_VectorAdd(float* A, float* B, float* result, float N, float scale) {
    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    global_vectorAdd << <blocksPerGrid, threadsPerBlock >> > (A, B, result, N, scale);
}

std::vector<float>& cu_utility::cuVectorAdd(const std::vector<float>& x,
    const std::vector<float>& b,
    std::vector<float>& result) {
    if (!(x.size() == b.size() && x.size() == result.size())) {
        std::cerr << "cuVectorAdd - Size does not match!";
        return result;
    }

    int N = x.size();  // Size of vectors
    size_t size = N * sizeof(float);

    // Allocate device memory
    float* d_x, * d_b, * d_r;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_r, size);

    // Copy data from host to device
    cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    global_vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_x, d_b, d_r, N);

    // Copy result from device to host
    cudaMemcpy(result.data(), d_r, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_r);

    return result;
}

std::vector<float>& cu_utility::cuSigmoid(std::vector<float>& x) {
    int N = x.size();  // Size of vectors
    size_t size = N * sizeof(float);

    // Allocate device memory
    float* d_x;
    cudaMalloc(&d_x, size);

    // Copy data from host to device
    cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    global_sigmoid << <blocksPerGrid, threadsPerBlock >> > (d_x, N);

    // Copy result from device to host
    cudaMemcpy(x.data(), d_x, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);

    return x;
}

std::vector<float>& cu_utility::cuDSigmoid(std::vector<float>& x) {
    int N = x.size();  // Size of vectors
    size_t size = N * sizeof(float);

    // Allocate device memory
    float* d_x;
    cudaMalloc(&d_x, size);

    // Copy data from host to device
    cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    global_d_sigmoid << <blocksPerGrid, threadsPerBlock >> > (d_x, N);

    // Copy result from device to host
    cudaMemcpy(x.data(), d_x, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);

    return x;
}

std::vector<float>& cu_utility::cuMatMulVector(
    const std::vector<std::vector<float>>& W, const std::vector<float>& x,
    std::vector<float>& result) {
    // Check Dims
    if (!(W[0].size() == x.size() && W.size() == result.size())) {
        std::cerr << "cuMatMulVector - Size does not match!";
        return result;
    }

    int M = result.size();
    int N = x.size();  // Size of vectors

    size_t sizeW = M * N * sizeof(float);
    size_t sizeX = N * sizeof(float);
    size_t sizeY = M * sizeof(float);

    // Allocate device memory
    float* d_W, * d_x, * d_y;
    cudaMalloc(&d_W, sizeW);
    cudaMalloc(&d_x, sizeX);
    cudaMalloc(&d_y, sizeY);

    // Copy data from host to device
    std::vector<float> W_flattened(M * N);
    for (int i = 0; i < M; i++) {
        std::copy(W[i].begin(), W[i].end(), W_flattened.begin() + i * N);
    }
    cudaMemcpy(d_W, W_flattened.data(), sizeW, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), sizeX, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    global_matMulVec << <blocksPerGrid, threadsPerBlock >> > (d_W, d_x, d_y, M, N);

    // Copy result from device to host
    cudaMemcpy(result.data(), d_y, sizeY, cudaMemcpyDeviceToHost);

    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_y);

    return result;
}

std::vector<float>& cu_utility::cuForwardLayer(
    const std::vector<std::vector<float>>& W, const std::vector<float>& b,
    const std::vector<float>& x, std::vector<float>& result) {
    int M = result.size();
    int N = x.size();

    size_t sizeW = M * N * sizeof(float);
    size_t sizeb = M * sizeof(float);
    size_t sizeX = N * sizeof(float);
    size_t sizeY = M * sizeof(float);

    // Allocate device memory
    float* d_W, * d_b, * d_x, * d_y;
    cudaMalloc(&d_W, sizeW);
    cudaMalloc(&d_b, sizeb);
    cudaMalloc(&d_x, sizeX);
    cudaMalloc(&d_y, sizeY);

    // Copy data from host to device
    std::vector<float> W_flattened(M * N);
    for (int i = 0; i < M; i++) {
        std::copy(W[i].begin(), W[i].end(), W_flattened.begin() + i * N);
    }
    cudaMemcpy(d_W, W_flattened.data(), sizeW, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), sizeX, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    global_forwardLayer << <blocksPerGrid, threadsPerBlock >> > (d_W, d_b, d_x, d_y,
        M, N);

    // Copy result from device to host
    cudaMemcpy(result.data(), d_y, sizeY, cudaMemcpyDeviceToHost);

    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_y);

    return result;
}

std::vector<float>& cu_utility::cuForwardLayerWithZs(const std::vector<std::vector<float>>& W,
	const std::vector<float>& b, const std::vector<float>& x, std::vector<float>& zsi, std::vector<float>& result)
{
    int M = result.size();
    int N = x.size();

    size_t sizeW = M * N * sizeof(float);
    size_t sizeb = M * sizeof(float);
    size_t sizeX = N * sizeof(float);
    size_t sizeY = M * sizeof(float);
    size_t sizeZsi = N * sizeof(float);

    // Allocate device memory
    float* d_W, * d_b, * d_x, * d_y, * d_zsi;
    cudaMalloc(&d_W, sizeW);
    cudaMalloc(&d_b, sizeb);
    cudaMalloc(&d_x, sizeX);
    cudaMalloc(&d_y, sizeY);
    cudaMalloc(&d_zsi, sizeZsi);

    // Copy data from host to device
    std::vector<float> W_flattened(M * N);
    for (int i = 0; i < M; i++) {
        std::copy(W[i].begin(), W[i].end(), W_flattened.begin() + i * N);
    }
    cudaMemcpy(d_W, W_flattened.data(), sizeW, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), sizeX, cudaMemcpyHostToDevice);
    cudaMemcpy(d_zsi, zsi.data(), sizeZsi, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    global_forwardLayer_zsi<< <blocksPerGrid, threadsPerBlock >> > (d_W, d_b, d_x, d_y,d_zsi,
        M, N);

    // Copy result from device to host
    cudaMemcpy(result.data(), d_y, sizeY, cudaMemcpyDeviceToHost);
    cudaMemcpy(zsi.data(), d_zsi, sizeZsi, cudaMemcpyDeviceToHost);

    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_zsi);

    return result;
}

void cu_utility::cuForwardLayerWithZs(const float* d_W, const float* d_b, const float* d_x,
	float* d_zsi, float* d_y, int M, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    global_forwardLayer_zsi << <blocksPerGrid, threadsPerBlock >> > (d_W, d_b, d_x, d_y, d_zsi,
        M, N);
    cudaDeviceSynchronize();
}


void cu_utility::cuBackwardOutputLayer(float* d_outActivation, float* d_inActivation,
    float* d_bias_output, float* d_weight_output,
    float* d_zsi, float* d_delta, const int* d_testLabel, int inSize, int outSize)
{
    // OutSize == 10 (the digits), inSize == 300 (layer[2])
    size_t f_size = sizeof(float);
    if(outSize != 10)
    {
        std::cerr << "Output layer should have an output size of 10!";
        return;
    }

    float* d_zstemp;
    cudaMalloc(&d_zstemp, f_size * outSize);
    cudaMemset(d_zstemp, 0, f_size * outSize);
	

    // Launch the kernel
    int threadsPerBlock = 256;

    int blocksPerGrid = (outSize + threadsPerBlock - 1) / threadsPerBlock;
    global_backwardLayer_output << <blocksPerGrid, threadsPerBlock >> > (d_outActivation, d_bias_output, d_zsi, d_zstemp, d_delta, d_testLabel, outSize);
    cudaDeviceSynchronize();
    blocksPerGrid = (inSize * outSize + threadsPerBlock - 1) / threadsPerBlock;
    global_outer_product << <blocksPerGrid, threadsPerBlock >> > (d_bias_output, d_inActivation, d_weight_output, outSize, inSize);

    cudaFree(d_zstemp);
}

void cu_utility::cuBackwardRegularLayer(float* d_inActivation, float* d_bias_output, float* d_weight_input,
	float* d_dWeight_output, float* d_zsi_in, float* d_zsi_out, float* d_delta_in, float* d_delta_out, int inSize,
	int outSize, int deltaSize)
{
    // TODO
    size_t f_size = sizeof(float);

    float* d_zstemp_o, * d_zstemp_i;
    cudaMalloc(&d_zstemp_o, f_size * outSize);
    cudaMalloc(&d_zstemp_i, f_size * inSize);
    cudaMemset(d_zstemp_o, 0, f_size * outSize);
    cudaMemset(d_zstemp_i, 0, f_size * inSize);
    // Launch the kernel
    int threadsPerBlock = 256;

    // in layer = numLayer-2-i in reference, out layer = numLayer-1-i in reference (for weight, bias and delta, not for zs / zs is allocated like activation)

    // activation_derivative
    int blocksPerGrid = (outSize + threadsPerBlock - 1) / threadsPerBlock;
    global_d_sigmoid_multiply_elementwise_delta << <blocksPerGrid, threadsPerBlock >> > (d_zsi_out, d_delta_out, d_delta_out, d_zstemp_o, outSize);
    cudaDeviceSynchronize();
    blocksPerGrid = (deltaSize * outSize + threadsPerBlock - 1) / threadsPerBlock;
    global_matTranMul << <blocksPerGrid, threadsPerBlock >> > (d_weight_input, d_delta_out, deltaSize, outSize, d_delta_in);
    // d_sigmoid_multiply
    cudaDeviceSynchronize();
    blocksPerGrid = (inSize + threadsPerBlock - 1) / threadsPerBlock;
    global_d_sigmoid_multiply_elementwise_delta << <blocksPerGrid, threadsPerBlock >> > (d_zsi_in, d_bias_output, d_delta_in, d_zstemp_i, inSize);
    // outer product
    cudaDeviceSynchronize();
    blocksPerGrid = (inSize * outSize + threadsPerBlock - 1) / threadsPerBlock;
    global_outer_product << <blocksPerGrid, threadsPerBlock >> > (d_bias_output, d_inActivation, d_dWeight_output, outSize, inSize);
    //cudaDeviceSynchronize();
    cudaFree(d_zstemp_o);
    cudaFree(d_zstemp_i);
}

float* cu_utility::copyDataToDevice(Matrix& X) {
    // flatten X
    int M = X.size();
    int N = X[0].size();
    size_t sizeX = M * N * sizeof(float);
    std::vector<float> X_flattened(M * N);

    for (int i = 0; i < M; i++) {
        std::copy(X[i].begin(), X[i].end(), X_flattened.begin() + i * N);
    }

    // copy to device
    float* d_X;
    cudaMalloc(&d_X, sizeX);
    cudaMemcpy(d_X, X_flattened.data(), sizeX, cudaMemcpyHostToDevice);

    return d_X;
}

int* cu_utility::copyDataToDevice(std::vector<int>& X) {
    // copy to device
    int* d_X;
    size_t sizeX = X.size() * sizeof(int);
    cudaMalloc(&d_X, sizeX);
    cudaMemcpy(d_X, X.data(), sizeX, cudaMemcpyHostToDevice);

    return d_X;
}

void cu_utility::testOuterProductAndTranspose(const std::vector<float>& a, const std::vector<float>& b,
	std::vector<float>& outer, std::vector<float>& transp)
{

    size_t matrix_len = a.size() * b.size();
    float* d_outer;
    float* d_transpose;
    float* d_a;
    float* d_b;
    cudaMalloc(&d_outer, matrix_len * sizeof(float));
    cudaMalloc(&d_transpose, matrix_len * sizeof(float));
    cudaMalloc(&d_a, a.size() * sizeof(float));
    cudaMalloc(&d_b, b.size() * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (b.size() + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);
    // Desired Output
    // Outer Product: 4 5 6 7 / 8 10 12 14 / 12 15 18 21
    // Transpose: 4 8 12 / 5 10 15 / 6 12 18 / 7 14 21
    global_test_kernel_matTran_outerProduct << <blocksPerGrid, threadsPerBlock >> >(d_a, d_b, a.size(), b.size(), d_outer, d_transpose);

    // 3 x 4 matrix multiply (1,0,2)
    float c[] = { 1,0,2 };
    std::vector<float> res;
    res.resize(4);

    float* d_c;
    float* d_res;
    cudaMalloc(&d_c, 3 * sizeof(float));
    cudaMalloc(&d_res, 4 * sizeof(float));
    cudaMemcpy(d_c, c, 3 * sizeof(float), cudaMemcpyHostToDevice);
    // Desired Output
    // 28 35 42 49
    global_matTranMul << <blocksPerGrid, threadsPerBlock >> > (d_outer, d_c, 3, 4, d_res); // Note that you have to input the TRANSPOSED LENGTH of the matrix!
    // Or TLDR: M = d_c.size(), N = d_res.size()

    outer.resize(matrix_len);
    transp.resize(matrix_len);

    cudaDeviceSynchronize();
    cudaMemcpy(outer.data(), d_outer, matrix_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(transp.data(), d_transpose, matrix_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res.data(), d_res, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    printVector(res, 1);

    
    cudaFree(d_outer);
    cudaFree(d_transpose);
    cudaFree(d_a);
    cudaFree(d_b);
}

void cu_utility::printVector(const std::vector<float>& v, const int& rowLength)
{
    size_t cutoff = v.size() > 300 ? 300 : v.size();
    for(int i = 0; i < cutoff; i++)
    {
	    if(rowLength > 0 && i % rowLength == 0)
	    {
            std::cout << '\n';
	    }
        std::cout << v[i] << " ";
    }
    std::cout << "\n";
}

void cu_utility::printVectorGPU(const float* d_v, int N, const int& rowLength) {
	size_t size = N * sizeof(float);
	std::vector<float> v(N);
	cudaMemcpy(v.data(), d_v, size, cudaMemcpyDeviceToHost);
	printVector(v, rowLength);
}

void cu_utility::printMatrixRowGPU(const float* d_v, int M, int N, int row, const int& rowLength) {
	size_t size = N * sizeof(float);
	std::vector<float> v(N);
	cudaMemcpy(v.data(), d_v + row * N, size, cudaMemcpyDeviceToHost);
	printVector(v, rowLength);
}


std::vector<std::vector<float>>& cu_utility::cuForward(
	const std::vector<float*> d_weights, const std::vector<float*> d_biases,
	const std::vector<float*> d_activations, const std::vector<int> layers,
	const std::vector<std::vector<float>>& X, std::vector<std::vector<float>>& result
	) {
    // flatten X
	int M = X.size();
	int N = X[0].size();
	size_t sizeX = M * N * sizeof(float);
	std::vector<float> X_flattened(M * N);

	for (int i = 0; i < M; i++) {
		std::copy(X[i].begin(), X[i].end(), X_flattened.begin() + i * N);
	}

	// copy to device
    float* d_X;
	cudaMalloc(&d_X, sizeX);
	cudaMemcpy(d_X, X_flattened.data(), sizeX, cudaMemcpyHostToDevice);

    // alloc memory for predictions
	float* d_predictions;
	cudaMalloc(&d_predictions, result.size() * result[0].size() * sizeof(float));

    // loop over examples in X
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	cudaDeviceSynchronize();
    for (int i = 0; i < M; i++) {
		float* d_x = d_X + i * N;
		global_forwardLayer << <blocksPerGrid, threadsPerBlock >> > (d_weights[0], d_biases[0], d_x, d_activations[1], layers[1], layers[0]);
        cudaDeviceSynchronize();
        for (int layer = 2; layer < layers.size(); layer++) {
            // Launch the kernel
            if (layer == layers.size() - 1) {
                // use predictions pointer
				global_forwardLayer << <blocksPerGrid, threadsPerBlock >> > (d_weights[layer - 1], d_biases[layer - 1], d_activations[layer - 1], d_predictions + i * result[0].size(), layers[layer], layers[layer - 1]);
            }
            else {
			    global_forwardLayer << <blocksPerGrid, threadsPerBlock >> > (d_weights[layer - 1], d_biases[layer - 1], d_activations[layer - 1], d_activations[layer], layers[layer], layers[layer - 1]);
            }
            cudaDeviceSynchronize();
        }   
    }

    // Copy result from device to host
    // predictionts flattened
	size_t sizePred = M * layers[layers.size() - 1] * sizeof(float);
	std::vector<float> predictions_flattened(M * layers[layers.size() - 1]);
	cudaMemcpy(predictions_flattened.data(), d_predictions, sizePred, cudaMemcpyDeviceToHost);

	// unflatten predictions
	for (int i = 0; i < M; i++) {
		std::copy(predictions_flattened.begin() + i * layers[layers.size() - 1], predictions_flattened.begin() + (i + 1) * layers[layers.size() - 1], result[i].begin());
	}

	cudaFree(d_X);
	cudaFree(d_predictions);

	return result;
}

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

int cu_utility::cuForwardBatch(
    const std::vector<float*> d_weights, const std::vector<float*> d_biases,
    const std::vector<float*> d_activations_batch, const std::vector<int> layers,
    const float* d_X,
	const int* d_Y,
    int numExamples,
    int batchSize
) {
    int N = 784;
    int implementation = 1; // 0 is default. 1 is default but separate global kernels, 2 is cublas, 3 is tensor cores.

    // alloc memory for predictions
	float* d_predictions;
	cudaMalloc(&d_predictions, numExamples * 10 * sizeof(float));

    dim3 blockDim(1, 32, 1);
    for (int i = 0; i < numExamples; i += batchSize) {
		const float* d_x = d_X + i * N;
		dim3 gridDim(batchSize, CEIL_DIV(300, blockDim.y), 1);

        for (int layer = 1; layer < layers.size(); layer++) {
			gridDim = dim3(batchSize, CEIL_DIV(layers[layer], blockDim.y), 1);

            const float* input = (layer == 1) ? d_x : d_activations_batch[layer - 1];
            float* output = (layer == layers.size() - 1) ? d_predictions + i * 10 : d_activations_batch[layer];

            switch (implementation) {
            case 0:
                global_forwardLayerBatch <<<gridDim, blockDim>>> (
                    d_weights[layer - 1],
                    d_biases[layer - 1],
                    input,
                    output,
                    layers[layer],
                    layers[layer - 1],
                    batchSize);
                break;
            case 1:
                // separate global kernels (Averages about 4ms slower than case 0)
				batched_forwardMatMul <<<gridDim, blockDim>>>(
					d_weights[layer - 1],
					input,
                    output,
					layers[layer],
					layers[layer - 1],
					batchSize);
                batched_addBiases <<<gridDim, blockDim >>> (
                    output,
                    d_biases[layer - 1],
                    output,
                    layers[layer],
                    batchSize);
                batched_sigmoid <<<gridDim, blockDim >>> (
                    output,
                    output,
                    layers[layer],
                    batchSize);
                break;
			case 2:
				// cublas
				break;
            case 3:
                // tensor cores
				break;
            default:
                std::cout << "Invalid Implementation" << std::endl;
            }
        }   
    }

    // evaluate preds on device
	int* d_numCorrect;
	cudaMalloc(&d_numCorrect, sizeof(int));
	cudaMemset(d_numCorrect, 0, sizeof(int));

	int threadsPerBlock = 128;
	int blocksPerGrid = CEIL_DIV(numExamples, threadsPerBlock);
	countNumCorrectPredictions << <blocksPerGrid, threadsPerBlock >> > (d_predictions, d_Y, d_numCorrect, numExamples, 10);

	int numCorrect;
	cudaMemcpy(&numCorrect, d_numCorrect, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_predictions);
	return numCorrect;
}



#define M 6
#define N 5
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

    static __inline__ void modify(cublasHandle_t handle, float* m, int ldm, int n, int p, int q, float alpha, float beta) {
        cublasSscal(handle, n - q, &alpha, &m[IDX2C(p, q, ldm)], ldm);
        cublasSscal(handle, ldm - p, &beta, &m[IDX2C(p, q, ldm)], 1);
    }



int cu_utility::testCuBlas() {
        cudaError_t cudaStat;
        cublasStatus_t stat;
        cublasHandle_t handle;
        int i, j;
        float* devPtrA;
        float* a = 0;
        a = (float*)malloc(M * N * sizeof(*a));
        if (!a) {
            printf("host memory allocation failed");
            return EXIT_FAILURE;
        }
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                a[IDX2C(i, j, M)] = (float)(i * N + j + 1);
            }
        }
        cudaStat = cudaMalloc((void**)&devPtrA, M * N * sizeof(*a));
        if (cudaStat != cudaSuccess) {
            printf("device memory allocation failed");
            free(a);
            return EXIT_FAILURE;
        }
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("CUBLAS initialization failed\n");
            free(a);
            cudaFree(devPtrA);
            return EXIT_FAILURE;
        }
        stat = cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("data download failed");
            free(a);
            cudaFree(devPtrA);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
        modify(handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
        stat = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("data upload failed");
            free(a);
            cudaFree(devPtrA);
            cublasDestroy(handle);
            return EXIT_FAILURE;
        }
        cudaFree(devPtrA);
        cublasDestroy(handle);
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                printf("%7.0f", a[IDX2C(i, j, M)]);
            }
            printf("\n");
        }
        free(a);
        return EXIT_SUCCESS;
}

