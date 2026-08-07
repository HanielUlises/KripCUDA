#pragma once

#include <string>

namespace kripcuda {

/// Host-only queries about the CUDA runtime. This header intentionally does not
/// include any CUDA header, so it can be used from plain C++ translation units.

/// Number of visible CUDA devices; 0 if the runtime is unavailable.
[[nodiscard]] int cudaDeviceCount() noexcept;

[[nodiscard]] inline bool hasCudaDevice() noexcept { return cudaDeviceCount() > 0; }

/// Human-readable name and compute capability of a device, or an empty string
/// if the device does not exist.
[[nodiscard]] std::string cudaDeviceDescription(int device = 0);

} // namespace kripcuda
