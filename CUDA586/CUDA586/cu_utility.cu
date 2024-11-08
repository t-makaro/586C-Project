#include "cu_utility.cuh"

// Device Kernels

__device__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
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

__device__ float sigmoid(float a) { return 1.0 / (1.0 + exp(-a)); }

__device__ void sigmoid(float* A, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        A[i] = sigmoid(A[i]);
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

// Global Kernels
__global__ void global_vectorAdd(const float* A, const float* B, float* C,
    int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
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

__global__ void global_test_kernel_matTran_outerProduct(const float* A, const float* B, int M, int N, float* resOuterProduct, float* resTranspose)
{
    outer_product(A, B, M, N, resOuterProduct);
    transpose(resOuterProduct, resTranspose, M, N); // Rearrange the input to N * M
}

cu_utility::cu_utility(/* args */) {}

cu_utility::~cu_utility() {}

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

    outer.resize(matrix_len);
    transp.resize(matrix_len);

    cudaDeviceSynchronize();
    cudaMemcpy(outer.data(), d_outer, matrix_len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(transp.data(), d_transpose, matrix_len * sizeof(float), cudaMemcpyDeviceToHost);

    
    cudaFree(d_outer);
    cudaFree(d_transpose);
    cudaFree(d_a);
    cudaFree(d_b);
}

void cu_utility::printVector(const std::vector<float>& v, const int& rowLength)
{
    for(int i = 0; i < v.size(); i++)
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

