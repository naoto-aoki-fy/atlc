#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <utility>

#include "format.hpp"

namespace atlc
{

    #define ATLC_CASE_RETURN(code) case code: return #code

    constexpr const char* get_filename_impl(const char* s, const char* last) noexcept
    {
        return (*s == '\0')
            ? last
            : ((*s == '/' || *s == '\\')
                    ? atlc::get_filename_impl(s + 1, s + 1)
                    : atlc::get_filename_impl(s + 1, last));
    }

    constexpr const char* get_filename(const char* s) noexcept
    {
        return atlc::get_filename_impl(s, s);
    }

    template <typename F>
    class Defer {
    public:
        explicit Defer(F f) : func_(std::move(f)) {}
        ~Defer() { func_(); }
    private:
        F func_;
    };

    template <typename F>
    Defer<F> make_defer(F f) { return Defer<F>(std::move(f)); }

    #define ATLC_CONCAT_INNER(a,b) a##b
    #define ATLC_CONCAT(a,b)       ATLC_CONCAT_INNER(a,b)
    #define ATLC_UNIQUE_NAME(base) ATLC_CONCAT(base,__LINE__)

    #define ATLC_DEFER_CODE(code)  auto ATLC_UNIQUE_NAME(_defer_) = atlc::make_defer([&] code);
    #define ATLC_DEFER_FUNC(func, ...) ATLC_DEFER_CODE({ (func)(__VA_ARGS__); })

    template <typename Func>
    auto check_zero(char const* const filename, int const lineno, char const* const funcname, Func func) -> decltype(func())
    {
        auto err = func();
        if (err != 0)
        {
            throw std::runtime_error(atlc::format("%s:%d:%s error:%d\n", filename, lineno, funcname, err));
        }
        return err;
    }

    #define ATLC_CHECK_ZERO(func, ...) atlc::check_zero(atlc::get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

    #define ATLC_DEFER_CHECK_ZERO(func, ...) ATLC_DEFER_CODE({ATLC_CHECK_ZERO(func, __VA_ARGS__);})


    template <typename Func>
    auto check_nonzero(char const* const filename, int const lineno, char const* const funcname, Func func) -> decltype(func())
    {
        auto err = func();
        if (err == 0)
        {
            throw std::runtime_error(atlc::format("%s:%d:%s error:%d\n", filename, lineno, funcname, err));
        }
        return err;
    }

    #define ATLC_CHECK_NONZERO(func, ...) atlc::check_nonzero(atlc::get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

    #define ATLC_DEFER_CHECK_NONZERO(func, ...) ATLC_DEFER_CODE({ATLC_CHECK_NONZERO(func, __VA_ARGS__);})


    template <typename CheckerFunc, typename Func>
    auto check_value(char const* const filename, int const lineno, CheckerFunc checker_func, char const* const funcname, Func func) -> decltype(func())
    {
        auto value = func();
        auto err = checker_func(value);
        if (err != 0)
        {
            throw std::runtime_error(atlc::format("%s:%d:%s error:%d\n", filename, lineno, funcname, err));
        }
        return value;
    }

    #define ATLC_CHECK_VALUE(checker_code, func, ...) atlc::check_value(get_filename(__FILE__), __LINE__, [&](auto arg)checker_code, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

}