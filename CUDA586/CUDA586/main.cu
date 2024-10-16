#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utility.h"
#include "cu_utility.cuh"
#include "cunn.cuh"

int main() {
     // NN 0: Init Neural Network
     std::vector<int> layers = { 784, 300, 300, 10 };
     CUNN nn(layers);

  //Util 0: Read Train Data
    vector<int> trainLabels;
    trainLabels.reserve(60000);
    std::cout << "Reading train data..." << std::endl;
    auto csvTrainData = utility::ReadDatasetCSV("../../data/train.csv", trainLabels);
    std::cout << "done." << std::endl;
    std::cout << "Training data size: " << csvTrainData.size() << "x"
            << csvTrainData[0].size() << std::endl;
    std::cout << "Training labels size: " << trainLabels.size() << std::endl;

  // Util 1: Read Test Data
    vector<int> testLabels;
    std::cout << "Reading test data..." << std::endl;
    auto csvTestData = utility::ReadDatasetCSV("../../data/test.csv", testLabels);
    std::cout << "done." << std::endl;
    std::cout << "Test data size: " << csvTestData.size() << "x"
            << csvTestData[0].size() << std::endl;
    std::cout << "Test labels size: " << testLabels.size() << std::endl;

  // Util 2: Read Weights and Biases
    auto biases_a1 = utility::ReadBias("../../data/biases_a1.csv");
    auto weights_a1 = utility::ReadWeight("../../data/weights_a1.csv");
    auto biases_a2 = utility::ReadBias("../../data/biases_a2.csv");
    auto weights_a2 = utility::ReadWeight("../../data/weights_a2.csv");
    auto biases_o = utility::ReadBias("../../data/biases_o.csv");
    auto weights_o = utility::ReadWeight("../../data/weights_o.csv");


    //// CU 2: Sigmoid Vec
    //int N = 512;
    //std::vector<float> t_gpusig(N, 0.4f);
    //std::vector<float> res_gpusig(N);

    //res_gpusig = cu_utility::cuSigmoid(t_gpusig);

    //std::cout << "Cuda Sigmoid Result: " << res_gpusig[256] << std::endl;

    //// CU 3: dSigmoid
    //std::vector<float> t_gpudsig(N, 0.4f);
    //std::vector<float> res_gpudsig(N);

    //res_gpudsig = cu_utility::cuDSigmoid(t_gpudsig);

    //std::cout << "Cuda dSigmoid Result: " << res_gpudsig[256] << std::endl;

    //// CU 1: Mat mul vector

    //int M = 2000;

    //std::vector<std::vector<float>> h_W;
    //h_W.reserve(M);
    //for (int i = 0; i < M; i++) {
    //    std::vector<float> Wi(N, 0.0f);
    //    if (i < N) Wi[i] = 1.0f;
    //    h_W.push_back(Wi);
    //}
    //std::vector<float> h_x(N, 1.0f);
    //std::vector<float> h_y(M);

    //std::cout << "Testing Mat Mul Vector..." << std::endl;
    //h_y = cu_utility::cuMatMulVector(h_W, h_x, h_y);

    // Verify the result
    //for (int i = 0; i < M; ++i) {
    //    if (i < N) {
    //        if (h_y[i] != 1.0f) {
    //            std::cerr << "Error at index " << i << ": " << h_y[i]
    //                << " != 1.0f\n";
    //            return -1;
    //        }
    //    }
    //    else {
    //        if (h_y[i] != 0.0f) {
    //            std::cerr << "Error at index " << i << ": " << h_y[i]
    //                << " != 0.0f\n";
    //            return -1;
    //        }
    //    }
    //}
    //std::cout << "All values are correct!\n";

    // NN 1: Copy Weights and Biases
    nn.copyWeights({ weights_a1, weights_a2, weights_o });
    nn.copyBiases({ biases_a1, biases_a2, biases_o });

    // NN 2: Forward Pass Training Set

	std::cout << "Evaluating on training set" << std::endl;
    nn.evaluate(csvTrainData, trainLabels);

    // NN 3: Forward Pass Test Set
    std::cout << "Evaluating on test set" << std::endl;
    nn.evaluate(csvTestData, testLabels);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
  return 0;
}
