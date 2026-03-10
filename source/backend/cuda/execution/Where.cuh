//
//  Where.cuh
//  MNN
//
//  Created by rainyl on 2024/03/10.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#ifndef Where_cuh
#define Where_cuh

#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

void WhereExecute(const void* input, const Tensor* inputTensor, int32_t* output, int total_count, int true_count, CUDABackend* backend);

} // namespace CUDA
} // namespace MNN

#endif /* Where_cuh */
