#include "kripcuda/cuda/launch.cuh"
#include "kripcuda/cuda/reduction.cuh"
#include "kripcuda/cuda/stream.cuh"
#include "kripcuda/verification/ctl_device.cuh"

#include <stdexcept>

namespace kripcuda {
namespace {

/// Every set-producing kernel below assigns one warp to one 32-bit word of the
/// characteristic function: lane l evaluates state 32w + l and __ballot_sync
/// assembles the word in a single instruction, so no atomics and no shared
/// memory are involved. Lanes whose state is out of range contribute a false
/// predicate, which is what keeps the padding bits of the last word at zero.
constexpr unsigned kSetBlock = 256;

[[nodiscard]] LaunchConfig setLaunch(std::uint32_t words) {
    return warpPerItemLaunch(words, kSetBlock);
}

__device__ __forceinline__ void writeBallot(SetWord* __restrict__ output, std::uint32_t word,
                                            bool predicate) {
    const unsigned ballot = __ballot_sync(0xffffffffu, predicate);
    if (threadIdx.x % kWarpSize == 0) {
        output[word] = static_cast<SetWord>(ballot);
    }
}

__global__ void atomKernel(DeviceKripkeView model, PropositionId proposition,
                           SetWord* __restrict__ output, std::uint32_t words) {
    const std::uint32_t lane = threadIdx.x % kWarpSize;
    const std::uint32_t warpsPerBlock = blockDim.x / kWarpSize;
    const auto word = static_cast<std::uint32_t>(blockIdx.x * warpsPerBlock + threadIdx.x / kWarpSize);
    if (word >= words) {
        return;
    }

    const StateId state = word * kSetWordBits + lane;
    const bool holds = state < model.stateCount && model.holds(state, proposition);
    writeBallot(output, word, holds);
}

/// Existential pre-image pre∃(X) = { s | ∃s'. R(s, s') ∧ s' ∈ X }, fused with
/// the surrounding set algebra of the CTL fixpoints:
///
///     next = base ∪ (mask ∩ pre∃(current))
///
/// Computing the pre-image as a gather over the forward relation — each state
/// inspecting its own successors — is what lets the framework avoid storing the
/// transposed relation at all: no scatter, no atomics, and the successor list
/// of a state is read exactly once per iteration.
__global__ void fixpointStepKernel(DeviceKripkeView model, const SetWord* __restrict__ base,
                                   const SetWord* __restrict__ mask,
                                   const SetWord* __restrict__ current,
                                   SetWord* __restrict__ next, std::uint32_t words,
                                   std::uint32_t* __restrict__ changed) {
    const std::uint32_t lane = threadIdx.x % kWarpSize;
    const std::uint32_t warpsPerBlock = blockDim.x / kWarpSize;
    const auto word = static_cast<std::uint32_t>(blockIdx.x * warpsPerBlock + threadIdx.x / kWarpSize);
    if (word >= words) {
        return;
    }

    const StateId state = word * kSetWordBits + lane;

    bool inPreImage = false;
    if (state < model.stateCount) {
        const EdgeIndex end = model.successorEnd(state);
        for (EdgeIndex edge = model.successorBegin(state); edge < end; ++edge) {
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

    const SetWord baseWord = base != nullptr ? base[word] : SetWord{0};
    const SetWord maskWord = mask != nullptr ? mask[word] : ~SetWord{0};
    const SetWord updated = baseWord | (maskWord & static_cast<SetWord>(ballot));

    next[word] = updated;
    if (updated != current[word]) {
        atomicOr(changed, 1U);
    }
}

/// EX has no fixpoint around it, so it skips the change detection entirely.
__global__ void preImageKernel(DeviceKripkeView model, const SetWord* __restrict__ current,
                               SetWord* __restrict__ next, std::uint32_t words) {
    const std::uint32_t lane = threadIdx.x % kWarpSize;
    const std::uint32_t warpsPerBlock = blockDim.x / kWarpSize;
    const auto word = static_cast<std::uint32_t>(blockIdx.x * warpsPerBlock + threadIdx.x / kWarpSize);
    if (word >= words) {
        return;
    }

    const StateId state = word * kSetWordBits + lane;

    bool inPreImage = false;
    if (state < model.stateCount) {
        const EdgeIndex end = model.successorEnd(state);
        for (EdgeIndex edge = model.successorBegin(state); edge < end; ++edge) {
            if (ConstDeviceStateSetView::containsOrEmpty(current, model.columns[edge])) {
                inPreImage = true;
                break;
            }
        }
    }

    writeBallot(next, word, inPreImage);
}

} // namespace

DeviceCtlEvaluator::DeviceCtlEvaluator(const DeviceKripke& model, cudaStream_t stream)
    : model_(model), stream_(stream), change_flag_(1) {}

DeviceStateSet DeviceCtlEvaluator::makeSet() { return DeviceStateSet(model_.stateCount()); }

bool DeviceCtlEvaluator::fixpointStep(const DeviceStateSet* base, const DeviceStateSet* mask,
                                      const DeviceStateSet& current, DeviceStateSet& next) {
    change_flag_.fillBytes(0, stream_);

    const std::uint32_t words = current.wordCount();
    const LaunchConfig launch = setLaunch(words);
    fixpointStepKernel<<<launch.grid, launch.block, 0, stream_>>>(
        model_.view(), base != nullptr ? base->words() : nullptr,
        mask != nullptr ? mask->words() : nullptr, current.words(), next.words(), words,
        change_flag_.data());
    KRIPCUDA_CHECK_LAST_ERROR();

    // One scalar per iteration is the entire host-device traffic of a fixpoint.
    std::uint32_t changed = 0;
    change_flag_.copyToHost(std::span<std::uint32_t>(&changed, 1), stream_);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream_));

    ++iterations_;
    return changed != 0;
}

DeviceStateSet& DeviceCtlEvaluator::evaluateInto(const ctl::Formula& formula) {
    const auto memoised = memo_.find(formula.identity());
    if (memoised != memo_.end()) {
        return memoised->second;
    }

    const std::uint32_t words = static_cast<std::uint32_t>(setWordsFor(model_.stateCount()));
    DeviceStateSet result = makeSet();

    switch (formula.kind()) {
    case ctl::Formula::Kind::False:
        result.clear(stream_);
        break;

    case ctl::Formula::Kind::Atom: {
        const LaunchConfig launch = setLaunch(words);
        atomKernel<<<launch.grid, launch.block, 0, stream_>>>(model_.view(), formula.proposition(),
                                                              result.words(), words);
        KRIPCUDA_CHECK_LAST_ERROR();
        break;
    }

    case ctl::Formula::Kind::Negation:
        result.assign(evaluateInto(formula.left()), stream_);
        result.complement(stream_);
        break;

    case ctl::Formula::Kind::Conjunction: {
        const DeviceStateSet& left = evaluateInto(formula.left());
        const DeviceStateSet& right = evaluateInto(formula.right());
        result.assign(left, stream_);
        result.intersectWith(right, stream_);
        break;
    }

    case ctl::Formula::Kind::Disjunction: {
        const DeviceStateSet& left = evaluateInto(formula.left());
        const DeviceStateSet& right = evaluateInto(formula.right());
        result.assign(left, stream_);
        result.unionWith(right, stream_);
        break;
    }

    case ctl::Formula::Kind::ExistsNext: {
        const DeviceStateSet& operand = evaluateInto(formula.left());
        const LaunchConfig launch = setLaunch(words);
        preImageKernel<<<launch.grid, launch.block, 0, stream_>>>(model_.view(), operand.words(),
                                                                  result.words(), words);
        KRIPCUDA_CHECK_LAST_ERROR();
        break;
    }

    case ctl::Formula::Kind::ExistsUntil: {
        // μX. ψ ∪ (φ ∩ pre∃(X)), from below.
        const DeviceStateSet& condition = evaluateInto(formula.left());
        const DeviceStateSet& target = evaluateInto(formula.right());
        DeviceStateSet next = makeSet();
        result.clear(stream_);
        while (fixpointStep(&target, &condition, result, next)) {
            result.swap(next);
        }
        break;
    }

    case ctl::Formula::Kind::ExistsGlobally: {
        // νX. φ ∩ pre∃(X), from above: the greatest fixpoint starts at ⟦φ⟧ and
        // shrinks, discarding states from which φ cannot be maintained.
        const DeviceStateSet& condition = evaluateInto(formula.left());
        DeviceStateSet next = makeSet();
        result.assign(condition, stream_);
        while (fixpointStep(nullptr, &condition, result, next)) {
            result.swap(next);
        }
        break;
    }
    }

    // References into the memo table must stay valid across the recursive calls
    // above; unordered_map guarantees reference stability for its mapped values.
    return memo_.emplace(formula.identity(), std::move(result)).first->second;
}

const DeviceStateSet& DeviceCtlEvaluator::evaluate(const ctl::Formula& formula) {
    return evaluateInto(formula);
}

CtlResult DeviceCtlEvaluator::check(const ctl::Formula& formula) {
    const DeviceStateSet& satisfying = evaluateInto(formula);

    CtlResult result;
    result.satisfying = satisfying.toHost(stream_);
    result.iterations = iterations_;
    result.holdsInAllInitialStates = true;
    for (const StateId initial : model_.initialStates()) {
        if (!result.satisfying.contains(initial)) {
            result.holdsInAllInitialStates = false;
            break;
        }
    }
    return result;
}

CtlResult checkCtlDevice(const DeviceKripke& model, const ctl::Formula& formula,
                         cudaStream_t stream) {
    DeviceCtlEvaluator evaluator(model, stream);
    return evaluator.check(formula);
}

CtlResult checkCtlDevice(const KripkeStructure& structure, const ctl::Formula& formula) {
    Stream stream;
    const DeviceKripke model(structure, stream);
    return checkCtlDevice(model, formula, stream);
}

} // namespace kripcuda
