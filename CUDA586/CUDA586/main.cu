#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utility.h"
#include "cu_utility.cuh"
#include "cunn.cuh"

#define TEST_FORWARD false
#define BACK_TEST true

int main() {
    // TEST CASES
    std::vector<float> t_a = {1.f, 2.f, 3.f};
    std::vector<float> t_b = {4.f, 5.f, 6.f, 7.f};
    std::vector<float> t_outer, t_transpose;
    cu_utility::printVector(t_a, 0);
    cu_utility::printVector(t_b, 4);

    cu_utility::testOuterProductAndTranspose(t_a, t_b, t_outer, t_transpose);
    cudaDeviceSynchronize();
    cu_utility::printVector(t_outer, 3);
    cu_utility::printVector(t_transpose, 4);

     // NN 0: Init Neural Network
     std::vector<int> layers = { 784, 300, 300, 10 };
     CUNN nn(layers);

  //Util 0: Read Train Data
#if TEST_FORWARD
     vector<int> trainLabels;
     trainLabels.reserve(60000);
     std::cout << "Reading train data..." << std::endl;
     auto csvTrainData = utility::ReadDatasetCSV("../../data/train.csv", trainLabels);
     std::cout << "done." << std::endl;
     std::cout << "Training data size: " << csvTrainData.size() << "x"
         << csvTrainData[0].size() << std::endl;
     std::cout << "Training labels size: " << trainLabels.size() << std::endl;
#endif

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

    // TNN 1: Forward Z Test

    CUNN tnn1(layers);
    CUNN tnn2(layers);
    tnn1.copyBiases({ biases_a1, biases_a2, biases_o });
    tnn1.copyWeights({ weights_a1, weights_a2, weights_o });
    tnn2.copyBiases({ biases_a1, biases_a2, biases_o });
    tnn2.copyWeights({ weights_a1, weights_a2, weights_o });
#if TEST_FORWARD
    //tnn2.copyParametersToDevice();
    //tnn1.testForwardZ(false, csvTestData[0]);
    //tnn2.testForwardZ(true, csvTestData[0]);
#endif

    // TNN 2: Backward Test
    CUNN tnn3(layers);
    CUNN tnn4(layers);
    if(BACK_TEST)
    {
        auto biases_a1_i = utility::ReadBias("../../data/biases_a1_init.csv");
        auto weights_a1_i = utility::ReadWeight("../../data/weights_a1_init.csv");
        auto biases_a2_i = utility::ReadBias("../../data/biases_a2_init.csv");
        auto weights_a2_i = utility::ReadWeight("../../data/weights_a2_init.csv");
        auto biases_o_i = utility::ReadBias("../../data/biases_o_init.csv");
        auto weights_o_i = utility::ReadWeight("../../data/weights_o_init.csv");

        tnn3.copyBiases({ biases_a1_i, biases_a2_i, biases_o_i });
        tnn3.copyWeights({ weights_a1_i, weights_a2_i, weights_o_i });
        tnn4.copyBiases({ biases_a1_i, biases_a2_i, biases_o_i });
        tnn4.copyWeights({ weights_a1_i, weights_a2_i, weights_o_i });

        tnn3.testBackwardOutputLayer(false, csvTestData[0], testLabels[0]);

        tnn4.copyParametersToDevice();
        tnn4.testBackwardOutputLayer(true, csvTestData[0], testLabels[0]);
    }
    




#if TEST_FORWARD
    // NN 1: Copy Weights and Biases and Data
    std::cout << "Copying Parameters and Data to the GPU" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    nn.copyWeights({ weights_a1, weights_a2, weights_o });
    nn.copyBiases({ biases_a1, biases_a2, biases_o });

    nn.copyParametersToDevice();

    float* d_trainData = cu_utility::copyDataToDevice(csvTrainData);
    int* d_trainLabels = cu_utility::copyDataToDevice(trainLabels);

    float* d_testData = cu_utility::copyDataToDevice(csvTestData);
    int* d_testLabels = cu_utility::copyDataToDevice(testLabels);

    int M_train = csvTrainData.size(); // num images
    int M_test = csvTestData.size(); // num images
    int N = csvTrainData[0].size(); // num pixels

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "\tdone." << std::endl;
    std::cout << "\tElapsed time: " << elapsed.count() << " seconds." << std::endl;

    int forwardBatchSize = 2400;
    nn.setBatchSizeDevice(forwardBatchSize);

    // NN 2: Forward Pass Training Set
    std::cout << "Evaluating on training set (Batched=" << forwardBatchSize << ")" << std::endl;
    nn.evaluate(d_trainData, d_trainLabels, M_train);

    // NN 3: Forward Pass Test Set
    std::cout << "Evaluating on test set (Batched=" << forwardBatchSize << ")" << std::endl;
	nn.evaluate(d_testData, d_testLabels, M_test);

    // NN 4: Training Set
    // nn.train(d_trainData, d_trainLabels, M_train, N, 1, 10, 0.1f);

    // Free thhe GPU memory
    cudaFree(d_trainData);
    cudaFree(d_trainLabels);
    cudaFree(d_testData);
    cudaFree(d_testLabels);
#endif
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
  return 0;
}
