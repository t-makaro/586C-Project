#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>

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
};