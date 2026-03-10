//
//  CudaWhere.cpp
//  MNN
//
//  Created by rainyl on 2024/03/10.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "backend/cuda/execution/CudaWhere.hpp"
#include "backend/cuda/execution/Where.cuh"

namespace MNN {
namespace CUDA {

ErrorCode CudaWhere::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input = inputs[0];
    auto output = outputs[0];

    int total_count = input->elementSize();
    int true_count = output->length(0); // output is [N, D]

    // If no elements, do nothing
    if (total_count == 0 || true_count == 0) {
        return NO_ERROR;
    }

    auto backend = static_cast<CUDABackend*>(this->backend());

    // Get device pointers
    const void* inputPtr = (const void*)input->deviceId();
    int32_t* outputPtr = (int32_t*)output->deviceId();

    WhereExecute(inputPtr, input, outputPtr, total_count, true_count, backend);

    return NO_ERROR;
}

class CudaWhereCreator : public CUDABackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CudaWhere(backend);
    }
};

static CUDACreatorRegister<CudaWhereCreator> __init(OpType_Where);

} // namespace CUDA
} // namespace MNN
