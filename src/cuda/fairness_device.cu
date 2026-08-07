#include "kripcuda/cuda/launch.cuh"
#include "kripcuda/cuda/stream.cuh"
#include "kripcuda/verification/fairness_device.cuh"

#include <stdexcept>
#include <utility>

namespace kripcuda {
namespace {

/// A component is a candidate for fairness as soon as it lies on a cycle; the
/// constraints then whittle the candidates down one at a time. The flag is
/// indexed by representative, so only the entries of representatives are ever
/// read.
__global__ void initFairKernel(DeviceKripkeView model, SccPartitionView partition,
                               unsigned char* __restrict__ fair) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < model.stateCount; state += stride) {
        fair[state] = partition.component[state] == state && partition.onCycle(model, state) ? 1U
                                                                                             : 0U;
    }
}

/// Marks the components that meet a constraint. Several states of a component
/// may write the same flag in the same pass; they all write the same value, so
/// the race needs no atomic.
__global__ void markConstraintKernel(SccPartitionView partition,
                                     const SetWord* __restrict__ constraint,
                                     unsigned char* __restrict__ hit) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < partition.stateCount; state += stride) {
        const StateId representative = partition.component[state];
        if (representative == kInvalidState) {
            continue;
        }
        if (((constraint[state / kSetWordBits] >> (state % kSetWordBits)) & SetWord{1}) != 0) {
            hit[representative] = 1U;
        }
    }
}

__global__ void intersectFlagsKernel(const unsigned char* __restrict__ hit, StateId stateCount,
                                     unsigned char* __restrict__ fair) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < stateCount; state += stride) {
        fair[state] = static_cast<unsigned char>(fair[state] & hit[state]);
    }
}

/// Lifts the per-component verdict back to a characteristic function, one warp
/// per word as everywhere else in the set layer.
__global__ void fairStatesKernel(SccPartitionView partition, const unsigned char* __restrict__ fair,
                                 SetWord* __restrict__ output, std::uint32_t words) {
    const std::uint32_t lane = threadIdx.x % kWarpSize;
    const std::uint32_t warpsPerBlock = blockDim.x / kWarpSize;
    const auto word =
        static_cast<std::uint32_t>(blockIdx.x * warpsPerBlock + threadIdx.x / kWarpSize);
    if (word >= words) {
        return;
    }

    const StateId state = word * kSetWordBits + lane;
    bool member = false;
    if (state < partition.stateCount) {
        const StateId representative = partition.component[state];
        member = representative != kInvalidState && fair[representative] != 0;
    }

    const unsigned ballot = __ballot_sync(0xffffffffu, member);
    if (lane == 0) {
        output[word] = static_cast<SetWord>(ballot);
    }
}

/// next = seed ∪ (region ∩ pre∃(current)) — backward reachability confined to a
/// region, computed as a gather over the forward relation so that no transpose
/// is needed and each successor list is read once per iteration.
__global__ void backwardStepKernel(DeviceKripkeView model, const SetWord* __restrict__ seed,
                                   const SetWord* __restrict__ region,
                                   const SetWord* __restrict__ current, SetWord* __restrict__ next,
                                   std::uint32_t words, std::uint32_t* __restrict__ changed) {
    const std::uint32_t lane = threadIdx.x % kWarpSize;
    const std::uint32_t warpsPerBlock = blockDim.x / kWarpSize;
    const auto word =
        static_cast<std::uint32_t>(blockIdx.x * warpsPerBlock + threadIdx.x / kWarpSize);
    if (word >= words) {
        return;
    }

    const StateId state = word * kSetWordBits + lane;

    bool inPreImage = false;
    if (state < model.stateCount) {
        for (EdgeIndex edge = model.successorBegin(state); edge < model.successorEnd(state);
             ++edge) {
            if (ConstDeviceStateSetView::containsOrEmpty(current, model.columns[edge])) {
                inPreImage = true;
                break;
            }
        }
    }

    const unsigned ballot = __ballot_sync(0xffffffffu, inPreImage);
    if (lane != 0) {
        return;
    }

    const SetWord updated = seed[word] | (region[word] & static_cast<SetWord>(ballot));
    next[word] = updated;
    if (updated != current[word]) {
        atomicOr(changed, 1U);
    }
}

struct FairAnalysis {
    DeviceStateSet fair;
    SccStatistics statistics;
};

[[nodiscard]] FairAnalysis analyseFairComponents(const DeviceKripke& model,
                                                 const DeviceStateSet& region,
                                                 const FairnessCondition& fairness,
                                                 const ReverseRelation& reverse,
                                                 cudaStream_t stream) {
    const StateId states = model.stateCount();
    if (region.stateCount() != states) {
        throw std::invalid_argument("fairness: region does not match the model");
    }

    DeviceStateSet result(states);
    if (states == 0) {
        return FairAnalysis{std::move(result), SccStatistics{}};
    }

    const SccPartition partition = computeSccDevice(model, reverse, &region, stream);
    const SccPartitionView partitionView = partition.view();
    const LaunchConfig launch = gridStrideLaunch(states);

    DeviceBuffer<unsigned char> fair(states);
    initFairKernel<<<launch.grid, launch.block, 0, stream>>>(model.view(), partitionView,
                                                             fair.data());
    KRIPCUDA_CHECK_LAST_ERROR();

    DeviceBuffer<unsigned char> hit(states);
    for (std::size_t index = 0; index < fairness.size(); ++index) {
        const DeviceStateSet& constraint = fairness[index];
        if (constraint.stateCount() != states) {
            throw std::invalid_argument("fairness: constraint does not match the model");
        }

        hit.fillBytes(0, stream);
        markConstraintKernel<<<launch.grid, launch.block, 0, stream>>>(
            partitionView, constraint.words(), hit.data());
        KRIPCUDA_CHECK_LAST_ERROR();

        intersectFlagsKernel<<<launch.grid, launch.block, 0, stream>>>(hit.data(), states,
                                                                       fair.data());
        KRIPCUDA_CHECK_LAST_ERROR();
    }

    const std::uint32_t words = result.wordCount();
    const LaunchConfig setLaunch = warpPerItemLaunch(words);
    fairStatesKernel<<<setLaunch.grid, setLaunch.block, 0, stream>>>(partitionView, fair.data(),
                                                                     result.words(), words);
    KRIPCUDA_CHECK_LAST_ERROR();
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

    return FairAnalysis{std::move(result), partition.statistics()};
}

/// μX. seed ∪ (region ∩ pre∃(X)), from below.
[[nodiscard]] DeviceStateSet backwardWithin(const DeviceKripke& model, const DeviceStateSet& region,
                                            const DeviceStateSet& seed, cudaStream_t stream) {
    const StateId states = model.stateCount();
    DeviceStateSet current(states);
    if (states == 0) {
        return current;
    }

    DeviceStateSet next(states);
    current.assign(seed, stream);

    DeviceBuffer<std::uint32_t> flag(1);
    const std::uint32_t words = current.wordCount();
    const LaunchConfig launch = warpPerItemLaunch(words);

    for (;;) {
        flag.fillBytes(0, stream);
        backwardStepKernel<<<launch.grid, launch.block, 0, stream>>>(
            model.view(), seed.words(), region.words(), current.words(), next.words(), words,
            flag.data());
        KRIPCUDA_CHECK_LAST_ERROR();

        std::uint32_t changed = 0;
        flag.copyToHost(std::span<std::uint32_t>(&changed, 1), stream);
        KRIPCUDA_CHECK(cudaStreamSynchronize(stream));
        if (changed == 0) {
            return current;
        }
        current.swap(next);
    }
}

} // namespace

DeviceStateSet fairComponentStates(const DeviceKripke& model, const DeviceStateSet& region,
                                   const FairnessCondition& fairness,
                                   const ReverseRelation& reverse, cudaStream_t stream) {
    return std::move(analyseFairComponents(model, region, fairness, reverse, stream).fair);
}

DeviceStateSet checkFairExistsGlobally(const DeviceKripke& model, const DeviceStateSet& condition,
                                       const FairnessCondition& fairness,
                                       const ReverseRelation& reverse, cudaStream_t stream) {
    FairAnalysis analysis = analyseFairComponents(model, condition, fairness, reverse, stream);
    return backwardWithin(model, condition, analysis.fair, stream);
}

FairCycleResult checkFairCycle(const DeviceKripke& model, const FairnessCondition& fairness,
                               cudaStream_t stream) {
    FairCycleResult result;
    const StateId states = model.stateCount();
    if (states == 0) {
        return result;
    }

    DeviceStateSet everywhere(states);
    everywhere.fillAll(stream);

    const ReverseRelation reverse(model, stream);
    FairAnalysis analysis = analyseFairComponents(model, everywhere, fairness, reverse, stream);
    result.scc = analysis.statistics;

    const DeviceStateSet witnesses = backwardWithin(model, everywhere, analysis.fair, stream);
    result.witnessCount = witnesses.count(stream);

    const StateSet host = witnesses.toHost(stream);
    for (const StateId initial : model.initialStates()) {
        if (host.contains(initial)) {
            result.holds = true;
            break;
        }
    }
    return result;
}

} // namespace kripcuda
