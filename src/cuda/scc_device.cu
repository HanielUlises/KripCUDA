#include "kripcuda/cuda/launch.cuh"
#include "kripcuda/cuda/reduction.cuh"
#include "kripcuda/cuda/stream.cuh"
#include "kripcuda/verification/scc_device.cuh"

#include <stdexcept>

namespace kripcuda {
namespace {

/// A null region denotes the whole state space, which is why this is not
/// ConstDeviceStateSetView::containsOrEmpty — there a null set is empty.
__device__ __forceinline__ bool inRegion(const SetWord* region, StateId state) {
    return region == nullptr ||
           ((region[state / kSetWordBits] >> (state % kSetWordBits)) & SetWord{1}) != 0;
}

/// A state is live while it belongs to the region and has not yet been assigned
/// to a component.
///
/// The component array is read here and written by the kernels below in the
/// same pass. That race is benign in both directions: reading a stale entry
/// leaves a state live for one more pass, and reading a fresh one only ever
/// removes a state that has genuinely been decided. Neither can put a state in
/// the wrong component, because both trimming and the backward closure only
/// ever grow the set of decided states.
__device__ __forceinline__ bool isLive(const SetWord* region, const StateId* component,
                                       StateId state) {
    return component[state] == kInvalidState && inRegion(region, state);
}

/// Removes the live states that cannot lie on a cycle of the live subgraph:
/// those with no live successor, and those with no live predecessor. Each one
/// is a singleton component, so trimming decides them outright. A state with a
/// self loop is its own live successor and predecessor and therefore survives.
__global__ void trimKernel(DeviceKripkeView model, ReverseRelationView reverse,
                           const SetWord* __restrict__ region, StateId* __restrict__ component,
                           std::uint32_t* __restrict__ changed) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < model.stateCount; state += stride) {
        if (!isLive(region, component, state)) {
            continue;
        }

        bool hasLiveSuccessor = false;
        for (EdgeIndex edge = model.successorBegin(state); edge < model.successorEnd(state);
             ++edge) {
            if (isLive(region, component, model.columns[edge])) {
                hasLiveSuccessor = true;
                break;
            }
        }

        bool hasLivePredecessor = false;
        if (hasLiveSuccessor) {
            for (EdgeIndex edge = reverse.predecessorBegin(state);
                 edge < reverse.predecessorEnd(state); ++edge) {
                if (isLive(region, component, reverse.columns[edge])) {
                    hasLivePredecessor = true;
                    break;
                }
            }
        }

        if (!hasLiveSuccessor || !hasLivePredecessor) {
            component[state] = state;
            atomicOr(changed, 1U);
        }
    }
}

__global__ void initColourKernel(const SetWord* __restrict__ region,
                                 const StateId* __restrict__ component, StateId stateCount,
                                 StateId* __restrict__ colour) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < stateCount; state += stride) {
        colour[state] = isLive(region, component, state) ? state : kInvalidState;
    }
}

/// Propagates each colour forward along the relation until the fixpoint
/// colour(v) = max { u live | u reaches v in the live subgraph }.
///
/// The step is a gather over the transpose rather than a scatter over the
/// forward relation: every state writes only its own entry, so the pass needs
/// no atomics on the colour array and its result does not depend on the order
/// in which blocks run. The iteration is a chaotic one — a state may read a
/// colour another block has already advanced — but the update is monotone, so
/// every schedule converges to the same least fixpoint.
__global__ void propagateColourKernel(ReverseRelationView reverse,
                                      const SetWord* __restrict__ region,
                                      const StateId* __restrict__ component,
                                      StateId* __restrict__ colour,
                                      std::uint32_t* __restrict__ changed) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < reverse.stateCount; state += stride) {
        if (!isLive(region, component, state)) {
            continue;
        }

        const StateId current = colour[state];
        StateId best = current;
        for (EdgeIndex edge = reverse.predecessorBegin(state); edge < reverse.predecessorEnd(state);
             ++edge) {
            const StateId predecessor = reverse.columns[edge];
            if (!isLive(region, component, predecessor)) {
                continue;
            }
            const StateId candidate = colour[predecessor];
            if (candidate != kInvalidState && candidate > best) {
                best = candidate;
            }
        }

        if (best != current) {
            colour[state] = best;
            atomicOr(changed, 1U);
        }
    }
}

/// A live state that carries its own colour is a root: no larger live state
/// reaches it, so its colour class contains its whole SCC.
__global__ void selectRootsKernel(const SetWord* __restrict__ region,
                                  const StateId* __restrict__ component,
                                  const StateId* __restrict__ colour, StateId stateCount,
                                  unsigned char* __restrict__ inComponent) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < stateCount; state += stride) {
        inComponent[state] =
            isLive(region, component, state) && colour[state] == state ? 1U : 0U;
    }
}

/// Backward closure of the roots inside their own colour class, which is the
/// intersection of the forward and backward sets of a root and therefore
/// exactly its SCC. Like the pre-image of the CTL fixpoints this is computed as
/// a gather over the forward relation — a state joins when one of its
/// successors already has — so the closure needs the transpose no more than the
/// model checker does.
__global__ void closeBackwardKernel(DeviceKripkeView model, const SetWord* __restrict__ region,
                                    const StateId* __restrict__ component,
                                    const StateId* __restrict__ colour,
                                    unsigned char* __restrict__ inComponent,
                                    std::uint32_t* __restrict__ changed) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < model.stateCount; state += stride) {
        if (inComponent[state] != 0 || !isLive(region, component, state)) {
            continue;
        }

        const StateId ownColour = colour[state];
        for (EdgeIndex edge = model.successorBegin(state); edge < model.successorEnd(state);
             ++edge) {
            const StateId successor = model.columns[edge];
            if (inComponent[successor] != 0 && colour[successor] == ownColour &&
                isLive(region, component, successor)) {
                inComponent[state] = 1U;
                atomicOr(changed, 1U);
                break;
            }
        }
    }
}

__global__ void assignKernel(const SetWord* __restrict__ region, const StateId* __restrict__ colour,
                             const unsigned char* __restrict__ inComponent, StateId stateCount,
                             StateId* __restrict__ component) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < stateCount; state += stride) {
        if (inComponent[state] != 0 && isLive(region, component, state)) {
            component[state] = colour[state];
        }
    }
}

__global__ void countLiveKernel(const SetWord* __restrict__ region,
                                const StateId* __restrict__ component, StateId stateCount,
                                std::uint32_t* __restrict__ result) {
    __shared__ std::uint32_t shared[kDefaultBlockSize / kWarpSize];

    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    std::uint32_t local = 0;
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < stateCount; state += stride) {
        local += isLive(region, component, state) ? 1U : 0U;
    }

    const std::uint32_t blockTotal = blockReduceSum(local, shared);
    if (threadIdx.x == 0 && blockTotal != 0) {
        atomicAdd(result, blockTotal);
    }
}

__global__ void componentSizeKernel(const StateId* __restrict__ component, StateId stateCount,
                                    StateId* __restrict__ size,
                                    std::uint32_t* __restrict__ componentCount) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < stateCount; state += stride) {
        const StateId representative = component[state];
        if (representative == kInvalidState) {
            continue;
        }
        atomicAdd(&size[representative], 1U);
        if (representative == state) {
            atomicAdd(componentCount, 1U);
        }
    }
}

__global__ void componentStatisticsKernel(DeviceKripkeView model,
                                          const StateId* __restrict__ component,
                                          const StateId* __restrict__ size, StateId stateCount,
                                          std::uint32_t* __restrict__ nontrivialCount,
                                          StateId* __restrict__ largest) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < stateCount; state += stride) {
        if (component[state] != state) {
            continue;
        }
        const StateId members = size[state];
        atomicMax(largest, members);
        if (members > 1 || model.hasTransition(state, state)) {
            atomicAdd(nontrivialCount, 1U);
        }
    }
}

/// One warp per word of the characteristic function, as everywhere else in the
/// set layer: lane l decides state 32w + l and __ballot_sync assembles the word.
__global__ void cyclicStatesKernel(DeviceKripkeView model, SccPartitionView partition,
                                   SetWord* __restrict__ output, std::uint32_t words) {
    const std::uint32_t lane = threadIdx.x % kWarpSize;
    const std::uint32_t warpsPerBlock = blockDim.x / kWarpSize;
    const auto word =
        static_cast<std::uint32_t>(blockIdx.x * warpsPerBlock + threadIdx.x / kWarpSize);
    if (word >= words) {
        return;
    }

    const StateId state = word * kSetWordBits + lane;
    const bool cyclic = state < model.stateCount && partition.onCycle(model, state);

    const unsigned ballot = __ballot_sync(0xffffffffu, cyclic);
    if (lane == 0) {
        output[word] = static_cast<SetWord>(ballot);
    }
}

/// Runs a kernel that reports through a device flag until it stops reporting a
/// change. One scalar transfer per iteration is the whole host-device traffic
/// of a fixpoint, the same bargain the CTL evaluator strikes.
template <typename Launch>
std::uint32_t iterateToFixpoint(DeviceBuffer<std::uint32_t>& flag, cudaStream_t stream,
                                Launch launch) {
    std::uint32_t iterations = 0;
    for (;;) {
        flag.fillBytes(0, stream);
        launch();
        KRIPCUDA_CHECK_LAST_ERROR();

        std::uint32_t changed = 0;
        flag.copyToHost(std::span<std::uint32_t>(&changed, 1), stream);
        KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

        ++iterations;
        if (changed == 0) {
            return iterations;
        }
    }
}

[[nodiscard]] std::uint32_t readScalar(const DeviceBuffer<std::uint32_t>& buffer,
                                       cudaStream_t stream) {
    std::uint32_t value = 0;
    buffer.copyToHost(std::span<std::uint32_t>(&value, 1), stream);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));
    return value;
}

} // namespace

SccPartition::SccPartition(DeviceBuffer<StateId> component, DeviceBuffer<StateId> size,
                           StateId stateCount, SccStatistics statistics)
    : component_(std::move(component)),
      size_(std::move(size)),
      state_count_(stateCount),
      statistics_(statistics) {}

DeviceStateSet SccPartition::cyclicStates(const DeviceKripke& model, cudaStream_t stream) const {
    if (model.stateCount() != state_count_) {
        throw std::invalid_argument("SccPartition::cyclicStates: model does not match the partition");
    }

    DeviceStateSet result(state_count_);
    if (state_count_ == 0) {
        return result;
    }

    const std::uint32_t words = result.wordCount();
    const LaunchConfig launch = warpPerItemLaunch(words);
    cyclicStatesKernel<<<launch.grid, launch.block, 0, stream>>>(model.view(), view(),
                                                                 result.words(), words);
    KRIPCUDA_CHECK_LAST_ERROR();
    return result;
}

SccPartition computeSccDevice(const DeviceKripke& model, const ReverseRelation& reverse,
                              const DeviceStateSet* region, cudaStream_t stream) {
    const StateId states = model.stateCount();
    if (reverse.stateCount() != states) {
        throw std::invalid_argument("computeSccDevice: transpose does not match the model");
    }
    if (region != nullptr && region->stateCount() != states) {
        throw std::invalid_argument("computeSccDevice: region does not match the model");
    }
    if (states == 0) {
        return SccPartition();
    }

    DeviceBuffer<StateId> component(states);
    component.fillBytes(0xFF, stream); // kInvalidState in every entry

    DeviceBuffer<StateId> colour(states);
    DeviceBuffer<unsigned char> inComponent(states);
    DeviceBuffer<std::uint32_t> flag(1);
    DeviceBuffer<std::uint32_t> counter(1);

    const SetWord* regionWords = region != nullptr ? region->words() : nullptr;
    const LaunchConfig launch = gridStrideLaunch(states);
    const DeviceKripkeView modelView = model.view();
    const ReverseRelationView reverseView = reverse.view();

    SccStatistics statistics;
    for (;;) {
        iterateToFixpoint(flag, stream, [&] {
            trimKernel<<<launch.grid, launch.block, 0, stream>>>(
                modelView, reverseView, regionWords, component.data(), flag.data());
        });

        counter.fillBytes(0, stream);
        countLiveKernel<<<launch.grid, launch.block, 0, stream>>>(regionWords, component.data(),
                                                                  states, counter.data());
        KRIPCUDA_CHECK_LAST_ERROR();
        if (readScalar(counter, stream) == 0) {
            break;
        }

        ++statistics.rounds;

        initColourKernel<<<launch.grid, launch.block, 0, stream>>>(regionWords, component.data(),
                                                                   states, colour.data());
        KRIPCUDA_CHECK_LAST_ERROR();

        iterateToFixpoint(flag, stream, [&] {
            propagateColourKernel<<<launch.grid, launch.block, 0, stream>>>(
                reverseView, regionWords, component.data(), colour.data(), flag.data());
        });

        selectRootsKernel<<<launch.grid, launch.block, 0, stream>>>(
            regionWords, component.data(), colour.data(), states, inComponent.data());
        KRIPCUDA_CHECK_LAST_ERROR();

        iterateToFixpoint(flag, stream, [&] {
            closeBackwardKernel<<<launch.grid, launch.block, 0, stream>>>(
                modelView, regionWords, component.data(), colour.data(), inComponent.data(),
                flag.data());
        });

        assignKernel<<<launch.grid, launch.block, 0, stream>>>(
            regionWords, colour.data(), inComponent.data(), states, component.data());
        KRIPCUDA_CHECK_LAST_ERROR();
    }

    DeviceBuffer<StateId> size(states);
    size.fillBytes(0, stream);
    counter.fillBytes(0, stream);
    componentSizeKernel<<<launch.grid, launch.block, 0, stream>>>(component.data(), states,
                                                                  size.data(), counter.data());
    KRIPCUDA_CHECK_LAST_ERROR();
    statistics.componentCount = readScalar(counter, stream);

    DeviceBuffer<StateId> largest(1);
    largest.fillBytes(0, stream);
    counter.fillBytes(0, stream);
    componentStatisticsKernel<<<launch.grid, launch.block, 0, stream>>>(
        modelView, component.data(), size.data(), states, counter.data(), largest.data());
    KRIPCUDA_CHECK_LAST_ERROR();
    statistics.nontrivialCount = readScalar(counter, stream);
    largest.copyToHost(std::span<StateId>(&statistics.largestComponent, 1), stream);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

    return SccPartition(std::move(component), std::move(size), states, statistics);
}

SccPartition computeSccDevice(const DeviceKripke& model, cudaStream_t stream) {
    const ReverseRelation reverse(model, stream);
    return computeSccDevice(model, reverse, nullptr, stream);
}

} // namespace kripcuda
