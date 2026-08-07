#pragma once

#include "kripcuda/cuda/device_buffer.cuh"

#include <cstdint>

namespace kripcuda {

/// Device-wide exclusive prefix sum over 32-bit counts, in the reduce-then-scan
/// form: a first pass reduces each tile, a single block scans the tile totals,
/// and a third pass scans each tile seeded by its offset. The input is read
/// twice, which is cheaper than the single-pass alternatives at this size and
/// keeps the implementation free of decoupled look-back and its memory-ordering
/// requirements.
///
/// Tile totals are accumulated in 64 bits so that an overflowing edge count is
/// detected rather than silently wrapping: exclusiveScan returns the exact
/// total and the caller decides whether it fits the index type.
///
/// The workspace grows monotonically, so reusing one instance across
/// constructions avoids reallocation.
class DeviceScan {
public:
    DeviceScan() = default;

    /// Writes the exclusive prefix sums of `input[0..count)` to `output` and
    /// returns the total. `output` may alias `input`.
    std::uint64_t exclusiveScan(const std::uint32_t* input, std::uint32_t* output,
                                std::size_t count, cudaStream_t stream = nullptr);

private:
    void reserve(std::size_t tiles);

    DeviceBuffer<std::uint64_t> tile_sums_;
    DeviceBuffer<std::uint64_t> tile_offsets_;
    DeviceBuffer<std::uint64_t> total_;
};

} // namespace kripcuda
