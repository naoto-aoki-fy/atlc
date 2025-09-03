// main.cu
// Example usage of atlc::block_reduce_sum_core

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { \
  cudaError_t _err = (expr); \
  if (_err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
    std::exit(1); \
  } \
} while(0)

#include <atlc/block_reduce_sum.cuh>

template <typename T, int N>
__global__ void block_reduce_sum_kernel(const T* __restrict__ in, T* __restrict__ out) {
    extern __shared__ T warp_partials[]; // size: sizeof(T) * N * warp_count
    const int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    const int tid_in_block = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int block_idx = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
    const int global_thread_id = block_idx * threads_per_block + tid_in_block;

    T vals[N];
#pragma unroll
    for (int k = 0; k < N; ++k) {
        vals[k] = in[global_thread_id * N + k];
    }

    atlc::block_reduce_sum_core<T>(vals, N, warp_partials);

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
#pragma unroll
        for (int k = 0; k < N; ++k) {
            out[block_idx * N + k] = vals[k];
        }
    }
}

template <typename T>
inline bool nearly_equal(T a, T b, T atol = static_cast<T>(1e-5), T rtol = static_cast<T>(1e-4)) {
    T diff = std::fabs(a - b);
    return diff <= atol + rtol * std::fabs(b);
}

template <typename T, int N>
bool run_case(dim3 grid, dim3 block, std::mt19937& rng, const char* label) {
    const int threads_per_block = block.x * block.y * block.z;
    const int total_threads = grid.x * grid.y * grid.z * threads_per_block;

    std::uniform_real_distribution<double> dist(static_cast<double>(-1.0), static_cast<double>(1.0));
    std::vector<T> h_in(static_cast<size_t>(total_threads) * N);
    for (auto& x : h_in) x = static_cast<T>(dist(rng));

    std::vector<T> h_ref(static_cast<size_t>(grid.x * grid.y * grid.z) * N, T(0));
    for (int b = 0; b < grid.x * grid.y * grid.z; ++b) {
        std::vector<double> acc(N, 0.0);
        for (int t = 0; t < threads_per_block; ++t) {
            const size_t gtid = static_cast<size_t>(b) * threads_per_block + t;
            for (int k = 0; k < N; ++k) {
                acc[k] += static_cast<double>(h_in[gtid * N + k]);
            }
        }
        for (int k = 0; k < N; ++k) {
            h_ref[b * N + k] = static_cast<T>(acc[k]);
        }
    }

    T *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(grid.x * grid.y * grid.z) * N * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, static_cast<size_t>(grid.x * grid.y * grid.z) * N * sizeof(T)));

    const int warp_count = (threads_per_block + 31) / 32;
    const size_t shmem_bytes = sizeof(T) * N * warp_count;

    block_reduce_sum_kernel<T, N><<<grid, block, shmem_bytes>>>(d_in, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> h_out(static_cast<size_t>(grid.x * grid.y * grid.z) * N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    bool ok = true;
    for (int b = 0; b < grid.x * grid.y * grid.z; ++b) {
        for (int k = 0; k < N; ++k) {
            const T got = h_out[b * N + k];
            const T exp = h_ref[b * N + k];
            if (!nearly_equal(got, exp)) {
                fprintf(stderr, "[FAIL] %s block %d, elem %d: got %.8f, exp %.8f\n",
                        label, b, k, static_cast<double>(got), static_cast<double>(exp));
                ok = false;
            }
        }
    }
    if (ok) {
        printf("[ OK ] %s passed (grid=(%d,%d,%d), block=(%d,%d,%d), N=%d)\n",
               label, grid.x, grid.y, grid.z, block.x, block.y, block.z, N);
    }
    return ok;
}

int main() {
    std::mt19937 rng(12345);
    bool all_ok = true;

    all_ok &= run_case<float, 4>(dim3(3,1,1), dim3(256,1,1), rng, "N=4, 1D block");
    all_ok &= run_case<float, 4>(dim3(3,1,1), dim3(64,2,2), rng, "N=4, 3D block");
    all_ok &= run_case<float, 4>(dim3(3,3,3), dim3(64,2,2), rng, "N=4, 3D block (multi-grid)");
    all_ok &= run_case<float, 1>(dim3(3,1,1), dim3(256,1,1), rng, "N=1, scalar");
    all_ok &= run_case<float, 7>(dim3(2,1,1), dim3(256,1,1), rng, "N=7, odd length");
    all_ok &= run_case<float, 4>(dim3(3,1,1), dim3(320,1,1), rng, "N=4, block=320");
    all_ok &= run_case<float, 4>(dim3(5,1,1), dim3(33,1,1), rng, "N=4, block=33");
    all_ok &= run_case<float, 7>(dim3(2,1,1), dim3(768,1,1), rng, "N=7, block=768");

    if (all_ok) {
        printf("\nAll tests passed ✅\n");
        return 0;
    } else {
        printf("\nSome tests FAILED ❌\n");
        return 1;
    }
}

