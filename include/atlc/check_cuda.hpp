#pragma once

// #include <cuda_runtime.h>
#include "check_x.hpp"
#include "format.hpp"
// #if defined(__CUDA_RUNTIME_H__)

namespace atlc {

    template <typename Func>
    void check_cuda(char const* const filename, int const lineno, char const* const funcname, Func func)
    {
        auto err = func();
        if (err != cudaSuccess)
        {
            char const* const error_string = cudaGetErrorString(err);
            throw std::runtime_error(atlc::format("%s:%d:%s error:%s\n", filename, lineno, funcname, error_string));
        }
    }

    #define ATLC_CHECK_CUDA(func, ...) atlc::check_cuda(atlc::get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

    #define ATLC_DEFER_CHECK_CUDA(func, ...) ATLC_DEFER_CODE({ATLC_CHECK_CUDA(func, __VA_ARGS__);})

}
// #endif /* __CUDA_RUNTIME_H__ */