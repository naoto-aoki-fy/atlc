#pragma once

#include "check_x.hpp"

// #if defined(_NVSHMEMX_H_)
namespace atlc {

template <typename Func>
void check_nvshmemx(char const* const filename, int const lineno, char const* const funcname, Func func)
{
    auto err = func();
    if (err != NVSHMEMX_SUCCESS)
    {
        throw std::runtime_error(atlc::format("%s:%d:%s error:%d\n", filename, lineno, funcname, err));
    }
}

#define ATLC_CHECK_NVSHMEMX(func, ...) atlc::check_nvshmem(atlc::get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){ return func(__VA_ARGS__); })

#define ATLC_DEFER_CHECK_NVSHMEMX(func, ...) ATLC_DEFER_CODE({ATLC_CHECK_NVSHMEMX(func, __VA_ARGS__);})

}

// #endif /* _NVSHMEMX_H_ */