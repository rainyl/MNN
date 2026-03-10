//
//  CudaWhere.hpp
//  MNN
//
//  Created by MNN on 2024/03/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CudaWhere_hpp
#define CudaWhere_hpp

#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

class CudaWhere : public Execution {
public:
    CudaWhere(Backend *b) : Execution(b) {}
    virtual ~CudaWhere() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
};

} // namespace CUDA
} // namespace MNN

#endif /* CudaWhere_hpp */
