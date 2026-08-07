#include "kripcuda/cuda/device_kripke.hpp"
#include "kripcuda/cuda/launch.hpp"
#include "kripcuda/cuda/reduction.hpp"
#include "kripcuda/cuda/stream.hpp"
#include "kripcuda/exploration/reachability.hpp"
#include "kripcuda/exploration/reachability_device.hpp"

#include <algorithm>

namespace kripcuda {
namespace {

/// Claims buffered in registers before the warp appends them to the frontier.
/// Beyond this many claims a lane falls back to its own atomic; eight covers
/// the out-degree of essentially every state in a concurrent model, where the
/// degree is the number of enabled transitions.
constexpr unsigned kLocalClaimCapacity = 8;

/// Out-degree above which a warp cooperates on a single state instead of one
/// thread expanding it alone. Below a full warp of successors, warp-per-state
/// expansion leaves lanes idle; above it, thread-per-state expansion serialises
/// a long successor list in one lane.
constexpr EdgeIndex kWarpExpansionThreshold = kWarpSize;

/// Appends per-lane claims with a single atomic per warp. Every lane of the
/// warp must reach this function — it is called after the expansion loop, not
/// inside it, so the ballot and shuffles below operate on a converged warp.
__device__ __forceinline__ void warpAppend(StateId* __restrict__ queue,
                                           std::uint32_t* __restrict__ queueSize,
                                           const StateId* local, std::uint32_t localCount) {
    std::uint32_t warpTotal = 0;
    const std::uint32_t prefix = warpExclusiveScan(localCount, warpTotal);
    if (warpTotal == 0) {
        return;
    }

    std::uint32_t base = 0;
    if (threadIdx.x % kWarpSize == 0) {
        base = atomicAdd(queueSize, warpTotal);
    }
    base = __shfl_sync(0xffffffffu, base, 0);

    for (std::uint32_t index = 0; index < localCount; ++index) {
        queue[base + prefix + index] = local[index];
    }
}

/// Claims `successor` for the next level. The compare-and-swap both reserves
/// the state and assigns its final BFS distance: every thread that could claim
/// it within a level writes the same value, so the level array does not depend
/// on which thread wins.
__device__ __forceinline__ bool claim(std::int32_t* __restrict__ levels, StateId successor,
                                      std::int32_t nextLevel) {
    return atomicCAS(&levels[successor], kUnreachable, nextLevel) == kUnreachable;
}

__global__ void seedInitialStatesKernel(const StateId* __restrict__ initialStates,
                                        std::uint32_t initialCount,
                                        std::int32_t* __restrict__ levels,
                                        StateId* __restrict__ frontier,
                                        std::uint32_t* __restrict__ frontierSize) {
    const auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);

    StateId local[kLocalClaimCapacity];
    std::uint32_t localCount = 0;
    if (index < initialCount) {
        const StateId state = initialStates[index];
        if (claim(levels, state, 0)) {
            local[localCount++] = state;
        }
    }
    warpAppend(frontier, frontierSize, local, localCount);
}

/// One thread per frontier state. Suitable while successor lists are short:
/// the whole list stays in one lane, so there is no intra-warp cooperation to
/// pay for, and the loads of a warp cover 32 different states.
__global__ void expandThreadPerStateKernel(DeviceKripkeView model,
                                           const StateId* __restrict__ frontier,
                                           std::uint32_t frontierSize, std::int32_t nextLevel,
                                           std::int32_t* __restrict__ levels,
                                           StateId* __restrict__ nextFrontier,
                                           std::uint32_t* __restrict__ nextFrontierSize) {
    const auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);

    StateId local[kLocalClaimCapacity];
    std::uint32_t localCount = 0;

    if (index < frontierSize) {
        const StateId state = frontier[index];
        const EdgeIndex end = model.successorEnd(state);
        for (EdgeIndex edge = model.successorBegin(state); edge < end; ++edge) {
            const StateId successor = model.columns[edge];
            if (!claim(levels, successor, nextLevel)) {
                continue;
            }
            if (localCount < kLocalClaimCapacity) {
                local[localCount++] = successor;
            } else {
                // Spill path for unusually branchy states: one atomic for a
                // claim that does not fit the register buffer.
                nextFrontier[atomicAdd(nextFrontierSize, 1U)] = successor;
            }
        }
    }

    warpAppend(nextFrontier, nextFrontierSize, local, localCount);
}

/// One warp per frontier state, lanes striding over the successor list. Chosen
/// for models with long successor lists, where a single lane would otherwise
/// serialise the expansion while its warp siblings idle.
__global__ void expandWarpPerStateKernel(DeviceKripkeView model,
                                         const StateId* __restrict__ frontier,
                                         std::uint32_t frontierSize, std::int32_t nextLevel,
                                         std::int32_t* __restrict__ levels,
                                         StateId* __restrict__ nextFrontier,
                                         std::uint32_t* __restrict__ nextFrontierSize) {
    const std::uint32_t lane = threadIdx.x % kWarpSize;
    const std::uint32_t warpsPerBlock = blockDim.x / kWarpSize;
    const auto warpIndex =
        static_cast<std::uint32_t>(blockIdx.x * warpsPerBlock + threadIdx.x / kWarpSize);

    StateId local[kLocalClaimCapacity];
    std::uint32_t localCount = 0;

    if (warpIndex < frontierSize) {
        const StateId state = frontier[warpIndex];
        const EdgeIndex begin = model.successorBegin(state);
        const EdgeIndex end = model.successorEnd(state);
        // Consecutive lanes read consecutive columns, so each pass over the
        // successor list is a coalesced load.
        for (EdgeIndex edge = begin + lane; edge < end; edge += kWarpSize) {
            const StateId successor = model.columns[edge];
            if (!claim(levels, successor, nextLevel)) {
                continue;
            }
            if (localCount < kLocalClaimCapacity) {
                local[localCount++] = successor;
            } else {
                nextFrontier[atomicAdd(nextFrontierSize, 1U)] = successor;
            }
        }
    }

    warpAppend(nextFrontier, nextFrontierSize, local, localCount);
}

} // namespace

ReachabilityResult computeReachabilityDevice(const DeviceKripke& model, cudaStream_t stream) {
    const StateId states = model.stateCount();

    ReachabilityResult result;
    result.levels.assign(states, kUnreachable);
    if (states == 0) {
        return result;
    }

    DeviceBuffer<std::int32_t> levels(states);
    levels.fillBytes(0xFF, stream); // -1 in two's complement

    // A frontier holds distinct states, so |S| entries are always sufficient.
    DeviceBuffer<StateId> frontier(states);
    DeviceBuffer<StateId> nextFrontier(states);
    DeviceBuffer<std::uint32_t> frontierSize(1);
    frontierSize.fillBytes(0, stream);

    const auto initialCount = static_cast<std::uint32_t>(model.initialStates().size());
    const LaunchConfig seedLaunch = linearLaunch(initialCount);
    seedInitialStatesKernel<<<seedLaunch.grid, seedLaunch.block, 0, stream>>>(
        model.initialStatesDevice().data(), initialCount, levels.data(), frontier.data(),
        frontierSize.data());
    KRIPCUDA_CHECK_LAST_ERROR();

    std::uint32_t currentSize = 0;
    frontierSize.copyToHost(std::span<std::uint32_t>(&currentSize, 1), stream);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

    const bool warpExpansion = model.statistics().maxOutDegree >= kWarpExpansionThreshold;

    StateId visited = currentSize;
    std::int32_t level = 0;
    while (currentSize > 0) {
        ++level;
        frontierSize.fillBytes(0, stream);

        if (warpExpansion) {
            const LaunchConfig launch = warpPerItemLaunch(currentSize);
            expandWarpPerStateKernel<<<launch.grid, launch.block, 0, stream>>>(
                model.view(), frontier.data(), currentSize, level, levels.data(),
                nextFrontier.data(), frontierSize.data());
        } else {
            const LaunchConfig launch = linearLaunch(currentSize);
            expandThreadPerStateKernel<<<launch.grid, launch.block, 0, stream>>>(
                model.view(), frontier.data(), currentSize, level, levels.data(),
                nextFrontier.data(), frontierSize.data());
        }
        KRIPCUDA_CHECK_LAST_ERROR();

        // One small device-to-host transfer per BFS level; the alternative
        // (dynamic parallelism or a persistent kernel) is not worth its cost at
        // the depths typical of explicit-state models.
        frontierSize.copyToHost(std::span<std::uint32_t>(&currentSize, 1), stream);
        KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

        visited += currentSize;
        std::swap(frontier, nextFrontier);
    }

    levels.copyToHost(result.levels, stream);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

    result.reachableCount = visited;
    result.maxLevel = visited == 0 ? kUnreachable : level - 1;
    return result;
}

ReachabilityResult computeReachabilityDevice(const KripkeStructure& structure) {
    Stream stream;
    const DeviceKripke model(structure, stream);
    return computeReachabilityDevice(model, stream);
}

} // namespace kripcuda
