#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__
 void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

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

// Use this function as playgrounds
int main() {
    int N = 1000; // Size of vectors

    // Allocate host memory
    std::vector<float> h_A(N, 1.0f); // Initialize with 1.0f
    std::vector<float> h_B(N, 2.0f); // Initialize with 2.0f
    std::vector<float> h_C(N);

    h_C = cu_utility::cuVectorAdd(h_A, h_B, h_C);

    // Verify the result
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != 3.0f) {
            std::cerr << "Error at index " << i << ": " << h_C[i] << " != 3.0f\n";
            return -1;
        }
    }

    std::cout << "All values are correct!\n";

    return 0;
}

