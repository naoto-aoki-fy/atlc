#pragma once

// #include <frida-gum.h>
#include "check_x.hpp"
#include "format.hpp"

namespace atlc {

    inline char const* const gumReplaceReturnToString(GumReplaceReturn ret) {
        switch(ret) {
            ATLC_CASE_RETURN(GUM_REPLACE_OK);
            ATLC_CASE_RETURN(GUM_REPLACE_WRONG_SIGNATURE);
            ATLC_CASE_RETURN(GUM_REPLACE_ALREADY_REPLACED);
            ATLC_CASE_RETURN(GUM_REPLACE_POLICY_VIOLATION);
            ATLC_CASE_RETURN(GUM_REPLACE_WRONG_TYPE);
            default: return NULL;
        }
    }

    template <typename Func>
    void check_frida_gum_replace(char const* const filename, int const lineno, char const* const funcname, Func func)
    {
        auto err = func();
        if (err != GUM_REPLACE_OK)
        {
            char const* const error_string = gumReplaceReturnToString(err);
            throw std::runtime_error(atlc::format("%s:%d:%s error:%s\n", filename, lineno, funcname, error_string));
        }
    }

    #define ATLC_CHECK_FRIDA_GUM_REPLACE(func, ...) atlc::check_frida_gum_replace(atlc::get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

}

