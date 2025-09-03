#pragma once

#include <cuda/std/complex>

namespace atlc {

    template<class Float>
    __forceinline__ __device__
    cuda::std::complex<Float> warp_shfl_down(unsigned mask, cuda::std::complex<Float> v, int off)
    {
        Float r = __shfl_down_sync(mask, v.real(), off);
        Float i = __shfl_down_sync(mask, v.imag(), off);
        return {r, i};
    }

    template <class T>
    __forceinline__ __device__
    T warp_shfl_down(unsigned mask, T v, int off)
    {
        return __shfl_down_sync(mask, v, off);
    }

    template <typename T>
    __forceinline__ __device__
    void warp_reduce_sum_vec(T* vals, int num_elements, unsigned mask)
    {
        #pragma unroll
        for (int off = warpSize / 2; off > 0; off >>= 1)
        {
            for (int k = 0; k < num_elements; ++k)
            {
                vals[k] += warp_shfl_down(mask, vals[k], off);
            }
        }
    }

    template <typename T>
    __forceinline__ __device__
    void block_reduce_sum_core(T* vals, int num_elements, T* warp_partials)
    {
        unsigned warp_mask = __activemask();
        warp_reduce_sum_vec<T>(vals, num_elements, warp_mask);

        size_t thread_num =
            (static_cast<size_t>(threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        int lane = static_cast<int>(thread_num) & (warpSize - 1);
        int wid  = static_cast<int>(thread_num) >> 5;

        const int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
        const int warp_count = (threads_per_block + warpSize - 1) / warpSize;

        if (lane == 0)
        {
            T* dst = &warp_partials[wid * num_elements];
            for (int k = 0; k < num_elements; ++k)
            {
                dst[k] = vals[k];
            }
        }
        __syncthreads();

        if (wid == 0)
        {
            for (int k = 0; k < num_elements; ++k)
            {
                vals[k] = T(0);
            }
            if (lane < warp_count)
            {
                for (int k = 0; k < num_elements; ++k)
                {
                    vals[k] = warp_partials[lane * num_elements + k];
                }
            }
            unsigned warp0_mask = __activemask();
            unsigned inter_mask = __ballot_sync(warp0_mask, lane < warp_count);
            warp_reduce_sum_vec<T>(vals, num_elements, inter_mask);
        }
    }

} // namespace atlc

