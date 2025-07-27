#pragma once

// #include <mpi.h>
#include "check_x.hpp"
#include "format.hpp"

// #if defined(MPI_COMM_WORLD)

namespace atlc {

    template <typename Func>
    void check_mpi(char const* const filename, int const lineno, char const* const funcname, Func func)
    {
        auto err = func();
        if (err != MPI_SUCCESS)
        {
            std::vector<char> error_string_vector(MPI_MAX_ERROR_STRING);
            int resultlen;
            MPI_Error_string(err, error_string_vector.data(), &resultlen);
            char const* const error_string = error_string_vector.data();
            throw std::runtime_error(atlc::format("%s:%d:%s error:%s\n", filename, lineno, funcname, error_string));
        }
    }

    #define ATLC_CHECK_MPI(func, ...) atlc::check_mpi(atlc::get_filename(__FILE__), __LINE__, #func "(" #__VA_ARGS__ ")", [&](){return func(__VA_ARGS__);})

    #define ATLC_DEFER_CHECK_MPI(func, ...) ATLC_DEFER_CODE({ATLC_CHECK_MPI(func, __VA_ARGS__);})

}

// #endif /* MPI_COMM_WORLD */