#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace kripcuda {

class CudaError : public std::runtime_error {
public:
    CudaError(cudaError_t status, const char* expression, const char* file, int line)
        : std::runtime_error(std::string(file) + ':' + std::to_string(line) + ": " + expression +
                             " failed: " + cudaGetErrorString(status)),
          status_(status) {}

    [[nodiscard]] cudaError_t status() const noexcept { return status_; }

private:
    cudaError_t status_;
};

namespace detail {

inline void checkCuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw CudaError(status, expression, file, line);
    }
}

} // namespace detail

} // namespace kripcuda

#define KRIPCUDA_CHECK(expression)                                                                 \
    ::kripcuda::detail::checkCuda((expression), #expression, __FILE__, __LINE__)

/// Validates both the launch configuration and any error left pending by an
/// earlier asynchronous operation on the current thread.
#define KRIPCUDA_CHECK_LAST_ERROR()                                                                \
    ::kripcuda::detail::checkCuda(cudaGetLastError(), "kernel launch", __FILE__, __LINE__)
