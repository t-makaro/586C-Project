#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// Kernels
__global__
 void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// APIs
class cu_utility
{
private:
    /* data */
public:
    cu_utility(/* args */);
    ~cu_utility();
    static std::vector<float>& cuVectorAdd(const std::vector<float> &x, const std::vector<float> &b, std::vector<float> &result);
};

cu_utility::cu_utility(/* args */)
{
}

cu_utility::~cu_utility()
{
}

std::vector<float> &cu_utility::cuVectorAdd(const std::vector<float> &x, const std::vector<float> &b, std::vector<float> &result)
{
    // TODO: insert return statement here
    if(!(x.size() == b.size() && x.size() == result.size())){
        std::cerr << "cuVectorAdd - Size does not match!";
        return result;
    }

    int N = x.size(); // Size of vectors
    size_t size = N * sizeof(float);

    // Allocate device memory
    float *d_x, *d_b, *d_r;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_r, size);

    // Copy data from host to device
    cudaMemcpy(d_x, x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_b, d_r, N);

    // Copy result from device to host
    cudaMemcpy(result.data(), d_r, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_b);
    cudaFree(d_r);

    return result;

}

