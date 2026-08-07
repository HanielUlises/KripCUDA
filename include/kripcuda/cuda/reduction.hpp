#pragma once

#include "kripcuda/cuda/launch.hpp"

#include <cstdint>

namespace kripcuda {

/// Device-side reduction primitives.
///
/// All warp-level helpers assume a full, converged warp: they are called from
/// code paths every lane reaches, which is why the shuffle mask is 0xffffffff
/// rather than __activemask(). Block-level helpers require the whole block to
/// participate, since they synchronise.

template <typename T>
__device__ __forceinline__ T warpReduceSum(T value) {
    for (unsigned offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename T>
__device__ __forceinline__ T warpReduceMax(T value) {
    for (unsigned offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const T other = __shfl_down_sync(0xffffffffu, value, offset);
        value = other > value ? other : value;
    }
    return value;
}

__device__ __forceinline__ unsigned warpReduceOr(unsigned value) {
    for (unsigned offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value |= __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

/// Inclusive prefix sum within a warp (Hillis-Steele over shuffles).
template <typename T>
__device__ __forceinline__ T warpInclusiveScan(T value) {
    const unsigned lane = threadIdx.x % kWarpSize;
    for (unsigned offset = 1; offset < kWarpSize; offset <<= 1) {
        const T other = __shfl_up_sync(0xffffffffu, value, offset);
        if (lane >= offset) {
            value += other;
        }
    }
    return value;
}

/// Exclusive prefix sum within a warp; `warpTotal` receives the warp aggregate
/// in every lane.
template <typename T>
__device__ __forceinline__ T warpExclusiveScan(T value, T& warpTotal) {
    const T inclusive = warpInclusiveScan(value);
    warpTotal = __shfl_sync(0xffffffffu, inclusive, kWarpSize - 1);
    return inclusive - value;
}

/// Block-wide reduction. Result is valid in thread 0 only. `shared` must hold
/// at least blockDim.x / kWarpSize elements.
template <typename T, typename WarpReduce>
__device__ __forceinline__ T blockReduce(T value, T* shared, WarpReduce warpReduce, T identity) {
    const unsigned lane = threadIdx.x % kWarpSize;
    const unsigned warp = threadIdx.x / kWarpSize;
    const unsigned warpCount = (blockDim.x + kWarpSize - 1) / kWarpSize;

    value = warpReduce(value);
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();

    if (warp != 0) {
        return identity;
    }
    T aggregate = lane < warpCount ? shared[lane] : identity;
    return warpReduce(aggregate);
}

template <typename T>
__device__ __forceinline__ T blockReduceSum(T value, T* shared) {
    return blockReduce(
        value, shared, [](T operand) { return warpReduceSum(operand); }, T{0});
}

template <typename T>
__device__ __forceinline__ T blockReduceMax(T value, T* shared, T identity) {
    return blockReduce(
        value, shared, [](T operand) { return warpReduceMax(operand); }, identity);
}

/// Block-wide exclusive prefix sum. Returns this thread's exclusive prefix and
/// writes the block aggregate to `blockAggregate` in every thread.
/// `shared` must hold at least blockDim.x / kWarpSize elements.
template <typename T>
__device__ __forceinline__ T blockExclusiveScan(T value, T* shared, T& blockAggregate) {
    const unsigned lane = threadIdx.x % kWarpSize;
    const unsigned warp = threadIdx.x / kWarpSize;
    const unsigned warpCount = (blockDim.x + kWarpSize - 1) / kWarpSize;

    const T inclusive = warpInclusiveScan(value);
    if (lane == kWarpSize - 1) {
        shared[warp] = inclusive;
    }
    __syncthreads();

    // A single warp scans the per-warp totals; warpCount never exceeds 32 for
    // the block sizes used here (at most 1024 threads).
    if (warp == 0) {
        const T warpTotal = lane < warpCount ? shared[lane] : T{0};
        const T scanned = warpInclusiveScan(warpTotal);
        if (lane < warpCount) {
            shared[lane] = scanned;
        }
    }
    __syncthreads();

    const T warpOffset = warp == 0 ? T{0} : shared[warp - 1];
    blockAggregate = shared[warpCount - 1];
    return warpOffset + inclusive - value;
}

} // namespace kripcuda
