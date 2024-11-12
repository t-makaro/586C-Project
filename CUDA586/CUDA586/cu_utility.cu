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
        }
        else {
            result[i] = 1.0f / (1.0f - last_activation[i] + FLT_EPSILON);
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
    vectorAdd(result, b, result, N);
    // activate
    sigmoid(result, M);
}

__global__ void global_forwardLayer_zsi(const float* W, const float* b, const float* A, float* result, float* zsi, int M, int N)
{
    // We need the intermediate value Zs in the backward pass.
    matMulVec(W, A, zsi, M, N);
    vectorAdd(zsi, b, zsi, N);
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

__global__ void global_forwardLayerBatch(const float* W, const float* b, const float* A, float* result, int M, int N, int batchSize)
{
	//matMul(W, A, result, M, N, batchSize);
	//matAddVec(result, b, M, batchSize);
	//sigmoidMat(result, M, batchSize);

	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

    // multiply, add, sigmoid
	if (i < M && j < batchSize) {
		float tmp = 0.0;
		for (int k = 0; k < N; k++) {
			tmp += W[i * N + k] * A[k * batchSize + j];
		}
		result[i * batchSize + j] = tmp + b[i];
		result[i * batchSize + j] = sigmoid(result[i * batchSize + j]);
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
	int outSize)
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
    blocksPerGrid = (inSize * outSize + threadsPerBlock - 1) / threadsPerBlock;
    global_matTranMul << <blocksPerGrid, threadsPerBlock >> > (d_weight_input, d_delta_out, outSize, inSize, d_delta_in);
    // d_sigmoid_multiply
    cudaDeviceSynchronize();
    blocksPerGrid = (inSize + threadsPerBlock - 1) / threadsPerBlock;
    global_d_sigmoid_multiply_elementwise_delta << <blocksPerGrid, threadsPerBlock >> > (d_zsi_in, d_bias_output, d_delta_in, d_zstemp_i, inSize);
    // outer product
    cudaDeviceSynchronize();
    blocksPerGrid = (inSize * outSize + threadsPerBlock - 1) / threadsPerBlock;
    global_outer_product << <blocksPerGrid, threadsPerBlock >> > (d_bias_output, d_inActivation, d_dWeight_output, outSize, inSize);
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

std::vector<std::vector<float>>& cu_utility::cuForwardBatch(
    const std::vector<float*> d_weights, const std::vector<float*> d_biases,
    const std::vector<float*> d_activations_batch, const std::vector<int> layers,
    const float* d_X,
    int batchSize,
    std::vector<std::vector<float>>& result
) {
	int M = result.size();
    int N = 784;

    // alloc memory for predictions
	float* d_predictions;
	cudaMalloc(&d_predictions, result.size() * result[0].size() * sizeof(float));

    dim3 blockDim(32, 32, 1);

	cudaDeviceSynchronize();
    for (int i = 0; i < M; i += batchSize) {
		const float* d_x = d_X + i * N;
		//dim3 gridDim = dim3(CEIL_DIV(784, 32), CEIL_DIV(784, 32), 1);
		dim3 gridDim(CEIL_DIV(batchSize, 32), CEIL_DIV(784, 32), 1);

		global_forwardLayerBatch << <gridDim, blockDim>> > (d_weights[0], d_biases[0], d_x, d_activations_batch[1], layers[1], layers[0], batchSize);
        cudaDeviceSynchronize();
        for (int layer = 2; layer < layers.size(); layer++) {
			gridDim = dim3(CEIL_DIV(batchSize, 32), CEIL_DIV(layers[layer], 32), 1);    
			//gridDim = dim3(CEIL_DIV(layers[layer], 32), CEIL_DIV(batchSize, 32), 1);
            // Launch the kernel
            if (layer == layers.size() - 1) {
                // use predictions pointer
				global_forwardLayerBatch << <gridDim, blockDim>> > (d_weights[layer - 1], d_biases[layer - 1], d_activations_batch[layer - 1], d_predictions + i * result[0].size(), layers[layer], layers[layer - 1], batchSize);
            }
            else {
				global_forwardLayerBatch << <gridDim, blockDim>> > (d_weights[layer - 1], d_biases[layer - 1], d_activations_batch[layer - 1], d_activations_batch[layer], layers[layer], layers[layer - 1], batchSize);
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

	cudaFree(d_predictions);

	return result;
}

