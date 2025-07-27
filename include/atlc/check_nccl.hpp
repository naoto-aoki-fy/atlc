#pragma once

// #include <nccl.h>
#include "check_x.hpp"

// #if defined(NCCL_H_)
namespace atlc {

    template <typename Func>
    void check_nccl(char const* const filename, int const lineno, char const* const funcname, Func func)
    {
        auto err = func();
        if (err != ncclSuccess)
        {
            char const* const error_string = ncclGetErrorString(err);
            throw std::runtime_error(atlc::format("%s:%d:%s error:%s\n", filename, lineno, funcname, error_string));
        }
    }

    #define ATLC_CHECK_NCCL(func, ...) atlc::check_nccl(atlc::get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

    #define ATLC_DEFER_CHECK_NCCL(func, ...) ATLC_DEFER_CODE({ATLC_CHECK_NCCL(func, __VA_ARGS__);})

}

// #endif /* NCCL_H_ */