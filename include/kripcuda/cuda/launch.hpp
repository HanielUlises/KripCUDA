#pragma once

#include "kripcuda/cuda/error.hpp"

#include <algorithm>
#include <cstddef>

namespace kripcuda {

inline constexpr unsigned kWarpSize = 32;
inline constexpr unsigned kDefaultBlockSize = 256;

/// Upper bound on the grid dimension used by grid-stride kernels. Beyond a few
/// thousand blocks a grid-stride loop is strictly better than more blocks: the
/// tail effect grows while occupancy no longer does.
inline constexpr unsigned kMaxGridStrideBlocks = 8192;

struct LaunchConfig {
    unsigned grid = 1;
    unsigned block = kDefaultBlockSize;
};

/// One thread per work item.
[[nodiscard]] inline LaunchConfig linearLaunch(std::size_t items,
                                               unsigned block = kDefaultBlockSize) {
    const std::size_t grid = (items + block - 1) / block;
    return LaunchConfig{static_cast<unsigned>(std::max<std::size_t>(grid, 1)), block};
}

/// One warp per work item; the block still holds `block` threads.
[[nodiscard]] inline LaunchConfig warpPerItemLaunch(std::size_t items,
                                                    unsigned block = kDefaultBlockSize) {
    const unsigned warpsPerBlock = block / kWarpSize;
    const std::size_t grid = (items + warpsPerBlock - 1) / warpsPerBlock;
    return LaunchConfig{static_cast<unsigned>(std::max<std::size_t>(grid, 1)), block};
}

/// Grid-stride launch: enough blocks to fill the machine, capped so that the
/// kernel loops instead of launching an unbounded grid.
[[nodiscard]] inline LaunchConfig gridStrideLaunch(std::size_t items,
                                                   unsigned block = kDefaultBlockSize,
                                                   unsigned maxBlocks = kMaxGridStrideBlocks) {
    const LaunchConfig linear = linearLaunch(items, block);
    return LaunchConfig{std::min(linear.grid, maxBlocks), block};
}

/// Block size that maximises occupancy for a kernel, as reported by the driver.
/// Used where the arithmetic intensity of a kernel is not known in advance;
/// the result is stable for a given device and kernel, so callers cache it.
template <typename Kernel>
[[nodiscard]] unsigned occupancyBlockSize(Kernel kernel, std::size_t dynamicSharedBytes = 0) {
    int minGridSize = 0;
    int blockSize = 0;
    KRIPCUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel,
                                                      dynamicSharedBytes, 0));
    return static_cast<unsigned>(blockSize);
}

} // namespace kripcuda
