#include "kripcuda/cuda/launch.hpp"
#include "kripcuda/cuda/reduction.hpp"
#include "kripcuda/cuda/scan.hpp"

#include <stdexcept>

namespace kripcuda {
namespace {

constexpr unsigned kScanBlock = 256;
constexpr unsigned kItemsPerThread = 4;
constexpr unsigned kTileSize = kScanBlock * kItemsPerThread;

__global__ void reduceTilesKernel(const std::uint32_t* __restrict__ input, std::size_t count,
                                  std::uint64_t* __restrict__ tileSums) {
    __shared__ std::uint64_t shared[kScanBlock / kWarpSize];

    const std::size_t tileBegin = static_cast<std::size_t>(blockIdx.x) * kTileSize;
    std::uint64_t local = 0;
    for (unsigned item = 0; item < kItemsPerThread; ++item) {
        const std::size_t index = tileBegin + item * kScanBlock + threadIdx.x;
        if (index < count) {
            local += input[index];
        }
    }

    const std::uint64_t total = blockReduceSum(local, shared);
    if (threadIdx.x == 0) {
        tileSums[blockIdx.x] = total;
    }
}

/// Single-block exclusive scan over the tile totals. The block walks the array
/// in chunks, carrying a running offset in shared memory, so it handles any
/// number of tiles without recursion.
__global__ void scanTileSumsKernel(const std::uint64_t* __restrict__ tileSums,
                                   std::uint64_t* __restrict__ tileOffsets, std::size_t tiles,
                                   std::uint64_t* __restrict__ total) {
    __shared__ std::uint64_t shared[kScanBlock / kWarpSize];
    __shared__ std::uint64_t carry;

    if (threadIdx.x == 0) {
        carry = 0;
    }
    __syncthreads();

    for (std::size_t base = 0; base < tiles; base += kScanBlock) {
        const std::size_t index = base + threadIdx.x;
        const std::uint64_t value = index < tiles ? tileSums[index] : 0;

        std::uint64_t aggregate = 0;
        const std::uint64_t prefix = blockExclusiveScan(value, shared, aggregate);
        if (index < tiles) {
            tileOffsets[index] = carry + prefix;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            carry += aggregate;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *total = carry;
    }
}

__global__ void scanTilesKernel(const std::uint32_t* __restrict__ input,
                                std::uint32_t* __restrict__ output, std::size_t count,
                                const std::uint64_t* __restrict__ tileOffsets) {
    __shared__ std::uint32_t shared[kScanBlock / kWarpSize];

    const std::size_t tileBegin = static_cast<std::size_t>(blockIdx.x) * kTileSize;
    auto running = static_cast<std::uint32_t>(tileOffsets[blockIdx.x]);

    // The tile is processed in kItemsPerThread block-wide passes so that the
    // per-pass carry stays in a register instead of a second shared array.
    for (unsigned item = 0; item < kItemsPerThread; ++item) {
        const std::size_t index = tileBegin + item * kScanBlock + threadIdx.x;
        const std::uint32_t value = index < count ? input[index] : 0;

        std::uint32_t aggregate = 0;
        const std::uint32_t prefix = blockExclusiveScan(value, shared, aggregate);
        if (index < count) {
            output[index] = running + prefix;
        }
        running += aggregate;
        __syncthreads();
    }
}

} // namespace

void DeviceScan::reserve(std::size_t tiles) {
    if (tile_sums_.size() < tiles) {
        tile_sums_ = DeviceBuffer<std::uint64_t>(tiles);
        tile_offsets_ = DeviceBuffer<std::uint64_t>(tiles);
    }
    if (total_.empty()) {
        total_ = DeviceBuffer<std::uint64_t>(1);
    }
}

std::uint64_t DeviceScan::exclusiveScan(const std::uint32_t* input, std::uint32_t* output,
                                        std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return 0;
    }

    const std::size_t tiles = (count + kTileSize - 1) / kTileSize;
    reserve(tiles);

    reduceTilesKernel<<<static_cast<unsigned>(tiles), kScanBlock, 0, stream>>>(input, count,
                                                                              tile_sums_.data());
    KRIPCUDA_CHECK_LAST_ERROR();

    scanTileSumsKernel<<<1, kScanBlock, 0, stream>>>(tile_sums_.data(), tile_offsets_.data(), tiles,
                                                     total_.data());
    KRIPCUDA_CHECK_LAST_ERROR();

    scanTilesKernel<<<static_cast<unsigned>(tiles), kScanBlock, 0, stream>>>(
        input, output, count, tile_offsets_.data());
    KRIPCUDA_CHECK_LAST_ERROR();

    std::uint64_t total = 0;
    total_.copyToHost(std::span<std::uint64_t>(&total, 1), stream);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));
    return total;
}

} // namespace kripcuda
