#include "kripcuda/cuda/launch.cuh"
#include "kripcuda/cuda/product.cuh"
#include "kripcuda/cuda/scan.cuh"

#include <limits>
#include <stdexcept>

namespace kripcuda {
namespace {

/// Index map of the product: (s₁, s₂) ↦ s₁·|S₂| + s₂. It is strictly monotone
/// in each component, which is what makes the merged successor lists produced
/// below come out sorted without a sorting pass.
__device__ __forceinline__ StateId encode(StateId left, StateId right, StateId rightStates) {
    return left * rightStates + right;
}

__global__ void interleavingDegreeKernel(DeviceKripkeView left, DeviceKripkeView right,
                                         StateId productStates,
                                         EdgeIndex* __restrict__ degrees) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < productStates; state += stride) {
        const StateId leftState = state / right.stateCount;
        const StateId rightState = state % right.stateCount;

        EdgeIndex degree = left.outDegree(leftState) + right.outDegree(rightState);
        // The only transition a product state can acquire twice: both
        // components staying put, which the two moves produce identically.
        if (left.hasTransition(leftState, leftState) &&
            right.hasTransition(rightState, rightState)) {
            degree -= 1;
        }
        degrees[state] = degree;
    }
}

__global__ void synchronousDegreeKernel(DeviceKripkeView left, DeviceKripkeView right,
                                        StateId productStates, EdgeIndex* __restrict__ degrees) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < productStates; state += stride) {
        const StateId leftState = state / right.stateCount;
        const StateId rightState = state % right.stateCount;
        degrees[state] = left.outDegree(leftState) * right.outDegree(rightState);
    }
}

/// Merges the two already sorted successor sequences of a product state. The
/// left move varies s₁ with s₂ fixed and the right move varies s₂ with s₁
/// fixed; both are ascending under the index map, so a linear merge emits them
/// in order and collapses the single possible duplicate.
__global__ void interleavingEdgeKernel(DeviceKripkeView left, DeviceKripkeView right,
                                       StateId productStates,
                                       const EdgeIndex* __restrict__ rowOffsets,
                                       StateId* __restrict__ columns) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < productStates; state += stride) {
        const StateId leftState = state / right.stateCount;
        const StateId rightState = state % right.stateCount;

        EdgeIndex leftEdge = left.successorBegin(leftState);
        const EdgeIndex leftEnd = left.successorEnd(leftState);
        EdgeIndex rightEdge = right.successorBegin(rightState);
        const EdgeIndex rightEnd = right.successorEnd(rightState);
        EdgeIndex out = rowOffsets[state];

        while (leftEdge < leftEnd && rightEdge < rightEnd) {
            const StateId fromLeft = encode(left.columns[leftEdge], rightState, right.stateCount);
            const StateId fromRight = encode(leftState, right.columns[rightEdge], right.stateCount);
            if (fromLeft < fromRight) {
                columns[out++] = fromLeft;
                ++leftEdge;
            } else if (fromRight < fromLeft) {
                columns[out++] = fromRight;
                ++rightEdge;
            } else {
                columns[out++] = fromLeft;
                ++leftEdge;
                ++rightEdge;
            }
        }
        while (leftEdge < leftEnd) {
            columns[out++] = encode(left.columns[leftEdge++], rightState, right.stateCount);
        }
        while (rightEdge < rightEnd) {
            columns[out++] = encode(leftState, right.columns[rightEdge++], right.stateCount);
        }
    }
}

__global__ void synchronousEdgeKernel(DeviceKripkeView left, DeviceKripkeView right,
                                      StateId productStates,
                                      const EdgeIndex* __restrict__ rowOffsets,
                                      StateId* __restrict__ columns) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < productStates; state += stride) {
        const StateId leftState = state / right.stateCount;
        const StateId rightState = state % right.stateCount;

        EdgeIndex out = rowOffsets[state];
        // Left successors in the outer loop, right in the inner: the emitted
        // indices are then ascending, since the map is monotone in s₁ first.
        for (EdgeIndex leftEdge = left.successorBegin(leftState);
             leftEdge < left.successorEnd(leftState); ++leftEdge) {
            for (EdgeIndex rightEdge = right.successorBegin(rightState);
                 rightEdge < right.successorEnd(rightState); ++rightEdge) {
                columns[out++] =
                    encode(left.columns[leftEdge], right.columns[rightEdge], right.stateCount);
            }
        }
    }
}

/// Concatenates the labellings: L(s₁, s₂) = L₁(s₁) ∪ (L₂(s₂) shifted above AP₁).
/// The shift is a bit-level shift of a multi-word bitset, so each output word
/// draws from two input words.
__global__ void productLabelKernel(DeviceKripkeView left, DeviceKripkeView right,
                                   StateId productStates, std::uint32_t leftPropositions,
                                   std::uint32_t productWords, LabelWord* __restrict__ labels) {
    const std::uint32_t wordShift = leftPropositions / kLabelWordBits;
    const std::uint32_t bitShift = leftPropositions % kLabelWordBits;

    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < productStates; state += stride) {
        const StateId leftState = state / right.stateCount;
        const StateId rightState = state % right.stateCount;

        const LabelWord* leftWords =
            left.labels + static_cast<std::size_t>(leftState) * left.labelWordsPerState;
        const LabelWord* rightWords =
            right.labels + static_cast<std::size_t>(rightState) * right.labelWordsPerState;
        LabelWord* out = labels + static_cast<std::size_t>(state) * productWords;

        for (std::uint32_t word = 0; word < productWords; ++word) {
            LabelWord value = word < left.labelWordsPerState ? leftWords[word] : LabelWord{0};

            if (word >= wordShift) {
                const std::uint32_t source = word - wordShift;
                if (source < right.labelWordsPerState) {
                    value |= rightWords[source] << bitShift;
                }
                // The high part of the previous word spills into this one only
                // when the shift is not word-aligned; shifting by 64 is
                // undefined, hence the guard rather than an unconditional shift.
                if (bitShift != 0 && source > 0 && source - 1 < right.labelWordsPerState) {
                    value |= rightWords[source - 1] >> (kLabelWordBits - bitShift);
                }
            }
            out[word] = value;
        }
    }
}

__global__ void writeTotalKernel(EdgeIndex* __restrict__ rowOffsets, StateId productStates,
                                 EdgeIndex total) {
    rowOffsets[productStates] = total;
}

} // namespace

DeviceKripke buildProduct(const DeviceKripke& left, const DeviceKripke& right, ProductKind kind,
                          cudaStream_t stream) {
    const auto leftStates = static_cast<std::uint64_t>(left.stateCount());
    const auto rightStates = static_cast<std::uint64_t>(right.stateCount());
    const std::uint64_t productStates = leftStates * rightStates;

    if (productStates == 0) {
        throw std::invalid_argument("buildProduct: both operands must be non-empty");
    }
    if (productStates > std::numeric_limits<StateId>::max()) {
        throw std::overflow_error("buildProduct: product state space exceeds the state index type");
    }

    const auto states = static_cast<StateId>(productStates);
    const std::uint32_t propositions = left.propositionCount() + right.propositionCount();
    const std::uint32_t productWords = labelWordsFor(propositions);

    DeviceBuffer<EdgeIndex> rowOffsets(static_cast<std::size_t>(states) + 1);
    DeviceBuffer<EdgeIndex> degrees(states);

    const LaunchConfig launch = gridStrideLaunch(states);
    if (kind == ProductKind::Interleaving) {
        interleavingDegreeKernel<<<launch.grid, launch.block, 0, stream>>>(
            left.view(), right.view(), states, degrees.data());
    } else {
        synchronousDegreeKernel<<<launch.grid, launch.block, 0, stream>>>(
            left.view(), right.view(), states, degrees.data());
    }
    KRIPCUDA_CHECK_LAST_ERROR();

    DeviceScan scan;
    const std::uint64_t transitions =
        scan.exclusiveScan(degrees.data(), rowOffsets.data(), states, stream);
    if (transitions > std::numeric_limits<EdgeIndex>::max()) {
        throw std::overflow_error("buildProduct: product relation exceeds the edge index type");
    }

    writeTotalKernel<<<1, 1, 0, stream>>>(rowOffsets.data(), states,
                                          static_cast<EdgeIndex>(transitions));
    KRIPCUDA_CHECK_LAST_ERROR();

    DeviceBuffer<StateId> columns(static_cast<std::size_t>(transitions));
    if (kind == ProductKind::Interleaving) {
        interleavingEdgeKernel<<<launch.grid, launch.block, 0, stream>>>(
            left.view(), right.view(), states, rowOffsets.data(), columns.data());
    } else {
        synchronousEdgeKernel<<<launch.grid, launch.block, 0, stream>>>(
            left.view(), right.view(), states, rowOffsets.data(), columns.data());
    }
    KRIPCUDA_CHECK_LAST_ERROR();

    DeviceBuffer<LabelWord> labels(static_cast<std::size_t>(states) * productWords);
    if (!labels.empty()) {
        productLabelKernel<<<launch.grid, launch.block, 0, stream>>>(
            left.view(), right.view(), states, left.propositionCount(), productWords,
            labels.data());
        KRIPCUDA_CHECK_LAST_ERROR();
    }

    std::vector<StateId> initialStates;
    initialStates.reserve(left.initialStates().size() * right.initialStates().size());
    for (const StateId leftInitial : left.initialStates()) {
        for (const StateId rightInitial : right.initialStates()) {
            initialStates.push_back(leftInitial * right.stateCount() + rightInitial);
        }
    }

    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

    return DeviceKripke(std::move(rowOffsets), std::move(columns), std::move(labels),
                        std::move(initialStates), states, propositions, stream);
}

DeviceKripke buildPower(const DeviceKripke& component, unsigned copies, ProductKind kind,
                        cudaStream_t stream) {
    if (copies == 0) {
        throw std::invalid_argument("buildPower: at least one copy is required");
    }

    DeviceKripke result = component.clone(stream);
    for (unsigned copy = 1; copy < copies; ++copy) {
        result = buildProduct(result, component, kind, stream);
    }
    return result;
}

} // namespace kripcuda
