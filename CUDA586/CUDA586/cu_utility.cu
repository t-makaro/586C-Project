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
