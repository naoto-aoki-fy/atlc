# A's Tiny Libraries for C++

A collection of single-header utilities that make day-to-day C++ and HPC
development easier.  The headers live under `include/atlc` and are designed to be
included piecemeal—you can cherry-pick only the helpers you need for a project.

Although the code was originally written for personal projects, the utilities
are intentionally small, dependency-free (beyond the platform SDKs they wrap),
and ready to be dropped into existing build systems.

## Highlights

- **Error handling wrappers** for CUDA, cuRAND, NCCL, NVSHMEM, MPI, and
  Frida-Gum via a consistent family of `ATLC_CHECK_*` macros.
- **Low-level CUDA helpers** such as a block-wide reduction primitive and a
  convenience wrapper for `cudaLaunchKernel`.
- **Header-only utilities** including a dynamic bitset container, integer
  `log2` helpers, `printf` format string helpers, and scope-based defers.
- **Self-contained examples** that demonstrate how to use the library with
  standard CMake-style toolchains (host C++, CUDA, and MPI).

## Repository layout

```
include/atlc/    # Library headers
examples/        # Small, buildable usage samples
LICENSE          # MIT license
README.md        # This file
```

Because the project is header-only, installation is as simple as adding the
`include` directory to your compiler's include path.

## Getting started

### Prerequisites

Only a standards-compliant C++11 compiler is required for the core headers.
Optional components have additional SDK requirements:

| Feature area                 | Additional requirements |
| ---------------------------- | ----------------------- |
| CUDA helpers & reductions    | CUDA toolkit            |
| cuRAND wrappers              | cuRAND headers          |
| NCCL wrappers                | NCCL headers            |
| NVSHMEM helpers              | NVSHMEM headers         |
| MPI utilities                | MPI implementation      |
| Frida-Gum wrappers           | Frida-Gum headers       |

### Using the headers

Add `include/` to your include path and include the relevant header.  For
example, to use the `dynamic_bitset`:

```cpp
#include <atlc/dynamic_bitset.hpp>

int main() {
    atlc::dynamic_bitset bits(10);
    bits.set(3);
    bits.flip(7);
    return static_cast<int>(bits.count());
}
```

### Building the examples

The `examples/` directory contains small programs that showcase the utilities.
They can be built with the provided `Makefile`:

```bash
cd examples
make                # builds all available examples
make clean          # removes the example binaries
```

Environment variables such as `CXX`, `MPICXX`, and `NVCC` can be overridden to
point to your preferred toolchains.

## Library components

Below is a quick reference of the most commonly used headers.  Refer to the
source for additional details.

| Header | Summary |
| ------ | ------- |
| `check_x.hpp` | Base utilities: `ATLC_CHECK_ZERO`, `ATLC_CHECK_NONZERO`, `ATLC_DEFER_CODE`, filename helpers, and the type-erased `atlc::Defer`. |
| `check_cuda.hpp`, `check_curand.hpp`, `check_nccl.hpp`, `check_mpi.hpp`, `check_nvshmemx.hpp`, `check_frida.hpp` | Domain-specific wrappers that throw a descriptive `std::runtime_error` when the wrapped API reports an error.  Each header provides `ATLC_CHECK_*` and `ATLC_DEFER_CHECK_*` macros that pair naturally with resource acquisition idioms. |
| `format.hpp` | Lightweight formatting helpers: `atlc::format` (a `printf`-style formatter returning `std::string`) and `ATLC_FORMAT` (a macro that mirrors `snprintf`). |
| `cuda.hpp` | A variadic template that forwards arguments to `cudaLaunchKernel` without having to spell out the `void**` array manually. |
| `block_reduce_sum.cuh` | Device-side helpers that implement warp-aware block reductions for scalars and `cuda::std::complex` values. |
| `dynamic_bitset.hpp` | A compact, resizable bitset with operations such as `set`, `reset`, `flip`, `count`, and `to_string`. |
| `log2_int.hpp` | Compile-time overloaded helpers for integer base-2 logarithms (`floor`, `ceil`, and exact). |
| `reorder.h` | Preprocessor utilities that make it easy to expand variadic macro arguments in alternating orders—handy for building `printf` statements without repeating variables. |
| `mpi.hpp` | Collects MPI ranks by hostname and assigns per-node indices, simplifying hybrid MPI + accelerator setups. |

## Examples

| Example | Description |
| ------- | ----------- |
| `printf.cpp` | Demonstrates `ATLC_REORDER` and friends to build type-safe `printf` invocations. |
| `dynamic_bitset.cpp` | Shows basic operations on `atlc::dynamic_bitset`. |
| `group_by_hostname.cpp` | Uses `atlc::group_by_hostname` to compute per-node ranks across an MPI job. |
| `block_reduce_sample.cu` | Runs a CUDA kernel that leverages `atlc::block_reduce_sum_core` to reduce vectors within a block and validates the results on the host. |

To run a specific example, build it and execute the generated binary (some
examples, such as the MPI or CUDA demos, must be run under the corresponding
runtime environment).

## License

The project is released under the MIT License.  See [`LICENSE`](LICENSE) for the
full text.
