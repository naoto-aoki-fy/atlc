#pragma once

#include <cuda_runtime.h>

namespace atlc {
    template<typename KernelType, typename... Args>
    cudaError_t cudaLaunchKernel(KernelType func, dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream, Args... args) {
        void* ptrs[] = {(void*)&args...};
        if (false) {
            func(args...);
        }
        return ::cudaLaunchKernel((void const*)func, gridDim, blockDim, ptrs, sharedMem, stream);
    }
}