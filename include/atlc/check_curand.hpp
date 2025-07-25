#pragma once

// #include <curand.h>
#include "check_x.hpp"

// #if defined(CURAND_H_)

namespace atlc {

    inline static char const* curandGetErrorString(curandStatus_t error) {
        switch (error) {
            ATLC_CASE_RETURN(CURAND_STATUS_SUCCESS);
            ATLC_CASE_RETURN(CURAND_STATUS_VERSION_MISMATCH);
            ATLC_CASE_RETURN(CURAND_STATUS_NOT_INITIALIZED);
            ATLC_CASE_RETURN(CURAND_STATUS_ALLOCATION_FAILED);
            ATLC_CASE_RETURN(CURAND_STATUS_TYPE_ERROR);
            ATLC_CASE_RETURN(CURAND_STATUS_OUT_OF_RANGE);
            ATLC_CASE_RETURN(CURAND_STATUS_LENGTH_NOT_MULTIPLE);
            ATLC_CASE_RETURN(CURAND_STATUS_DOUBLE_PRECISION_REQUIRED);
            ATLC_CASE_RETURN(CURAND_STATUS_LAUNCH_FAILURE);
            ATLC_CASE_RETURN(CURAND_STATUS_PREEXISTING_FAILURE);
            ATLC_CASE_RETURN(CURAND_STATUS_INITIALIZATION_FAILED);
            ATLC_CASE_RETURN(CURAND_STATUS_ARCH_MISMATCH);
            ATLC_CASE_RETURN(CURAND_STATUS_INTERNAL_ERROR);
            default: return NULL;
        }
        return "<unknown>";
    }

    template <typename Func>
    void check_curand(char const* const filename, int const lineno, char const* const funcname, Func func)
    {
        auto err = func();
        if (err != CURAND_STATUS_SUCCESS)
        {
            // fprintf(stderr, "[debug] %s:%d call:%s error:%s\n", filename, lineno, funcname, curandGetErrorString(err));

            std::vector<char> strbuf(1);
            char const* const error_string = curandGetErrorString(err);

            auto printf_lambda = [=](char* strbuf, size_t buf_length){ return snprintf(strbuf, buf_length, "%s:%d:%s error:%s\n", filename, lineno, funcname, error_string); };

            int str_length = printf_lambda(strbuf.data(), strbuf.size());
            strbuf.resize(str_length + 1);
            str_length = printf_lambda(strbuf.data(), strbuf.size() + 1);

            throw std::runtime_error(strbuf.data());
        }
    }

    #define ATLC_CHECK_CURAND(func, ...) atlc::check_curand(atlc::get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

    #define ATLC_DEFER_CHECK_CURAND(func, ...) ATLC_DEFER_CODE({ATLC_CHECK_CURAND(func, __VA_ARGS__);})

}

// #endif /* CURAND_H_ */