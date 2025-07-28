#pragma once

#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>

#define ATLC_FORMAT(...) ([&]() -> std::string { \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wformat\"") \
    int const len = std::snprintf(NULL, 0, __VA_ARGS__); \
    _Pragma("GCC diagnostic pop") \
    if (len < 0) { throw std::runtime_error("snprintf failed"); } \
    std::string result(static_cast<std::size_t>(len) + 1, '\0'); \
    std::snprintf(&result[0], result.size(), __VA_ARGS__); \
    result.resize(static_cast<std::size_t>(len)); \
    return result; \
})()

namespace atlc
{
    __attribute__ ((__format__(printf, 1, 2)))
    std::string format(const char* fmt, ...)
    {
        if (!fmt) return {};
    
        va_list args;
        va_start(args, fmt);
    
        va_list args_copy;
        va_copy(args_copy, args);
        const int len = std::vsnprintf(nullptr, 0, fmt, args_copy);
        va_end(args_copy);
    
        if (len < 0) {
            va_end(args);
            throw std::runtime_error("vsnprintf failed");
        }
    
        std::string result(static_cast<std::size_t>(len) + 1, '\0');
    
        std::vsnprintf(&result[0], result.size(), fmt, args);
        va_end(args);
    
        result.resize(static_cast<std::size_t>(len));
        return result;
    }
    
}