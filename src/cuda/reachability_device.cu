#include "kripcuda/cuda/device_kripke.hpp"
#include "kripcuda/exploration/reachability.hpp"

#include <algorithm>

namespace kripcuda {
namespace {

constexpr int kBlockSize = 256;

__global__ void seedInitialStatesKernel(const StateId* __restrict__ initialStates,
                                        std::uint32_t initialCount,
                                        std::int32_t* __restrict__ levels,
                                        StateId* __restrict__ frontier,
                                        std::uint32_t* __restrict__ frontierSize) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= initialCount) {
        return;
    }
    const StateId state = initialStates[index];
    // Duplicate initial states are already removed by the builder, but the CAS
    // keeps the invariant "appended exactly once" independent of that.
    if (atomicCAS(&levels[state], kUnreachable, 0) == kUnreachable) {
        frontier[atomicAdd(frontierSize, 1u)] = state;
    }
}

/// One thread per frontier state. Successor lists in a Kripke structure are
/// typically short and of similar length (bounded by the number of enabled
/// transitions), so per-thread expansion keeps the kernel simple without
/// serious load imbalance; skewed models would call for warp-level expansion.
__global__ void expandFrontierKernel(DeviceKripkeView model,
                                     const StateId* __restrict__ frontier,
                                     std::uint32_t frontierSize,
                                     std::int32_t nextLevel,
                                     std::int32_t* __restrict__ levels,
                                     StateId* __restrict__ nextFrontier,
                                     std::uint32_t* __restrict__ nextFrontierSize) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= frontierSize) {
        return;
    }
    const StateId state = frontier[index];
    const EdgeIndex end = model.successorEnd(state);
    for (EdgeIndex edge = model.successorBegin(state); edge < end; ++edge) {
        const StateId successor = model.columns[edge];
        // The CAS both claims the state and assigns its final BFS level: every
        // claim within a level writes the same value, so the result does not
        // depend on the winner.
        if (atomicCAS(&levels[successor], kUnreachable, nextLevel) == kUnreachable) {
            nextFrontier[atomicAdd(nextFrontierSize, 1u)] = successor;
        }
    }
}

std::uint32_t gridFor(std::uint32_t threads) {
    return (threads + kBlockSize - 1) / kBlockSize;
}

} // namespace

ReachabilityResult computeReachabilityDevice(const KripkeStructure& structure) {
    const StateId states = structure.stateCount();

    ReachabilityResult result;
    result.levels.assign(states, kUnreachable);
    if (states == 0) {
        return result;
    }

    cudaStream_t stream = nullptr;
    KRIPCUDA_CHECK(cudaStreamCreate(&stream));

    // The stream is destroyed on every exit path, including the exception one.
    struct StreamGuard {
        cudaStream_t stream;
        ~StreamGuard() { static_cast<void>(cudaStreamDestroy(stream)); }
    } guard{stream};

    const DeviceKripke model(structure, stream);

    DeviceBuffer<std::int32_t> levels(states);
    levels.fillBytes(0xFF, stream); // -1 in two's complement

    // A frontier holds distinct states, so |S| entries are always sufficient.
    DeviceBuffer<StateId> frontier(states);
    DeviceBuffer<StateId> nextFrontier(states);
    DeviceBuffer<std::uint32_t> frontierSize(1);
    frontierSize.fillBytes(0, stream);

    const auto initialCount = static_cast<std::uint32_t>(model.initialStates().size());
    seedInitialStatesKernel<<<gridFor(initialCount), kBlockSize, 0, stream>>>(
        model.initialStatesDevice().data(), initialCount, levels.data(), frontier.data(),
        frontierSize.data());
    KRIPCUDA_CHECK_LAST_ERROR();

    std::uint32_t currentSize = 0;
    frontierSize.copyToHost(std::span<std::uint32_t>(&currentSize, 1), stream);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

    StateId visited = currentSize;
    std::int32_t level = 0;
    while (currentSize > 0) {
        ++level;
        frontierSize.fillBytes(0, stream);
        expandFrontierKernel<<<gridFor(currentSize), kBlockSize, 0, stream>>>(
            model.view(), frontier.data(), currentSize, level, levels.data(), nextFrontier.data(),
            frontierSize.data());
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
    result.maxLevel = level - 1;
    if (visited == 0) {
        result.maxLevel = kUnreachable;
    }
    return result;
}

} // namespace kripcuda
