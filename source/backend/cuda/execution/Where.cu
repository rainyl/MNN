//
//  Where.cu
//  MNN
//
//  Created by rainyl on 2024/03/10.
//  Copyright © 2026, Alibaba Group Holding Limited
//

#include "Where.cuh"
#include <cub/cub.cuh>
#include "backend/cuda/core/CUDABackend.hpp"

namespace MNN {
namespace CUDA {

template <typename T>
struct GreaterThanZero {
    __host__ __device__ bool operator()(const T& x) const {
        return x > (T)0;
    }
};

__global__ void WhereCoordsKernel(const int* linear_indices, int32_t* output, int count, int dimensions,
                                  const int* strides) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    int index = linear_indices[idx];
    for (int j = 0; j < dimensions; j++) {
        int stride = strides[j];
        int result = stride == 0 ? index : index / stride;
        index = index - result * stride;
        output[idx * dimensions + j] = result;
    }
}

template <typename T>
void WhereImpl(const void* input, const Tensor* inputTensor, int32_t* output, int total_count, int true_count, CUDABackend* backend) {
    auto runtime = backend->getCUDARuntime();
    cudaStream_t stream = 0;

    const T* inputData = static_cast<const T*>(input);

    // 1. Select indices
    cub::CountingInputIterator<int> counting_iter(0);
    cub::TransformInputIterator<bool, GreaterThanZero<T>, const T*> flags_iter(inputData, GreaterThanZero<T>());

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Determine temporary device storage requirements
    // We need a place to store the number of selected items, even if we know it.
    // CUB requires a pointer to device memory for num_selected_out.
    // We can use a small buffer from the pool.
    auto pool = backend->getBufferPool();
    auto num_selected_buffer = pool->alloc(sizeof(int));
    int* d_num_selected_out = (int*)num_selected_buffer.ptr();

    // We also need a buffer for the output linear indices.
    // This is temporary, size = true_count * sizeof(int).
    // If true_count is 0, we can skip.
    if (true_count == 0) {
        pool->free(num_selected_buffer);
        return;
    }

    auto indices_buffer = pool->alloc(true_count * sizeof(int));
    int* d_indices = (int*)indices_buffer.ptr();

    // Get temp storage size
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting_iter, flags_iter, d_indices, d_num_selected_out, total_count, stream);

    // Allocate temp storage
    auto temp_buffer = pool->alloc(temp_storage_bytes);
    d_temp_storage = temp_buffer.ptr();

    // Run selection
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, counting_iter, flags_iter, d_indices, d_num_selected_out, total_count, stream);

    // 2. Convert linear indices to coordinates
    // Prepare strides
    // We need to copy strides to device
    auto& ib = inputTensor->buffer();
    int dimensions = ib.dimensions;
    std::vector<int> strides_host(dimensions);
    for (int i = 0; i < dimensions; ++i) {
        strides_host[i] = ib.dim[i].stride;
    }

    auto strides_buffer = pool->alloc(dimensions * sizeof(int));
    cudaMemcpyAsync(strides_buffer.ptr(), strides_host.data(), dimensions * sizeof(int), cudaMemcpyHostToDevice, stream);

    int block_size = 256;
    int grid_size = (true_count + block_size - 1) / block_size;

    WhereCoordsKernel<<<grid_size, block_size, 0, stream>>>(d_indices, output, true_count, dimensions, (int*)strides_buffer.ptr());

    // Free buffers
    pool->free(temp_buffer);
    pool->free(indices_buffer);
    pool->free(num_selected_buffer);
    pool->free(strides_buffer);
}

void WhereExecute(const void* input, const Tensor* inputTensor, int32_t* output, int total_count, int true_count, CUDABackend* backend) {
    auto type = inputTensor->getType();
    if (type == halide_type_of<float>()) {
        WhereImpl<float>(input, inputTensor, output, total_count, true_count, backend);
    } else if (type == halide_type_of<int32_t>()) {
        WhereImpl<int32_t>(input, inputTensor, output, total_count, true_count, backend);
    } else if (type == halide_type_of<uint8_t>()) {
        WhereImpl<uint8_t>(input, inputTensor, output, total_count, true_count, backend);
    } else {
        MNN_ERROR("CUDA Where op doesn't support type %d\n", type.code);
    }
}

} // namespace CUDA
} // namespace MNN
