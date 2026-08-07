#pragma once

#include "kripcuda/cuda/device_kripke.hpp"
#include "kripcuda/exploration/reachability.hpp"

namespace kripcuda {

/// Reachability over a model that already lives on the device. This is the
/// entry point to use for a model produced on the GPU — a product, say — since
/// it never moves the transition relation across the bus.
[[nodiscard]] ReachabilityResult computeReachabilityDevice(const DeviceKripke& model,
                                                           cudaStream_t stream = nullptr);

} // namespace kripcuda
