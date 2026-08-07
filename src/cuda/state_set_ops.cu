#include "kripcuda/cuda/device_state_set.cuh"
#include "kripcuda/cuda/launch.cuh"
#include "kripcuda/cuda/reduction.cuh"

#include <stdexcept>

namespace kripcuda {
namespace {

enum class BinaryOp { Assign, Union, Intersection, Difference };

/// Elementwise word algebra. One kernel with a compile-time operation keeps the
/// four set operations to a single instantiation each, with no branch in the
/// inner loop.
template <BinaryOp Op>
__global__ void binaryOpKernel(SetWord* __restrict__ target, const SetWord* __restrict__ source,
                               std::uint32_t words) {
    const auto stride = static_cast<std::uint32_t>(blockDim.x * gridDim.x);
    for (auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
         index < words; index += stride) {
        const SetWord operand = source[index];
        if constexpr (Op == BinaryOp::Assign) {
            target[index] = operand;
        } else if constexpr (Op == BinaryOp::Union) {
            target[index] |= operand;
        } else if constexpr (Op == BinaryOp::Intersection) {
            target[index] &= operand;
        } else {
            target[index] &= ~operand;
        }
    }
}

__global__ void fillKernel(SetWord* __restrict__ target, SetWord value, SetWord lastWordMask,
                           std::uint32_t words) {
    const auto stride = static_cast<std::uint32_t>(blockDim.x * gridDim.x);
    for (auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
         index < words; index += stride) {
        target[index] = index + 1 == words ? (value & lastWordMask) : value;
    }
}

__global__ void complementKernel(SetWord* __restrict__ target, SetWord lastWordMask,
                                 std::uint32_t words) {
    const auto stride = static_cast<std::uint32_t>(blockDim.x * gridDim.x);
    for (auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
         index < words; index += stride) {
        const SetWord complemented = ~target[index];
        target[index] = index + 1 == words ? (complemented & lastWordMask) : complemented;
    }
}

__global__ void cardinalityKernel(const SetWord* __restrict__ words, std::uint32_t wordCount,
                                  std::uint32_t* __restrict__ result) {
    __shared__ std::uint32_t shared[kDefaultBlockSize / kWarpSize];

    const auto stride = static_cast<std::uint32_t>(blockDim.x * gridDim.x);
    std::uint32_t local = 0;
    for (auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
         index < wordCount; index += stride) {
        local += static_cast<std::uint32_t>(__popc(words[index]));
    }

    const std::uint32_t blockTotal = blockReduceSum(local, shared);
    if (threadIdx.x == 0 && blockTotal != 0) {
        // Integer addition is associative, so the reduced value does not depend
        // on the order in which blocks arrive.
        atomicAdd(result, blockTotal);
    }
}

enum class PredicateOp { Difference, NotSubset };

template <PredicateOp Op>
__global__ void predicateKernel(const SetWord* __restrict__ left,
                                const SetWord* __restrict__ right, std::uint32_t words,
                                std::uint32_t* __restrict__ result) {
    const auto stride = static_cast<std::uint32_t>(blockDim.x * gridDim.x);
    std::uint32_t witness = 0;
    for (auto index = static_cast<std::uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
         index < words; index += stride) {
        if constexpr (Op == PredicateOp::Difference) {
            witness |= left[index] ^ right[index];
        } else {
            witness |= left[index] & ~right[index];
        }
    }

    // Only the existence of a witness matters, so the warp collapses its lanes
    // before a single lane touches global memory.
    const unsigned any = warpReduceOr(witness != 0 ? 1U : 0U);
    if (threadIdx.x % kWarpSize == 0 && any != 0) {
        atomicOr(result, 1U);
    }
}

} // namespace

DeviceStateSet::DeviceStateSet(StateId stateCount)
    : words_(setWordsFor(stateCount)), state_count_(stateCount), scalar_(1) {
    clear();
}

DeviceStateSet::DeviceStateSet(const StateSet& host, cudaStream_t stream)
    : words_(setWordsFor(host.stateCount())), state_count_(host.stateCount()), scalar_(1) {
    words_.copyFromHost(host.words(), stream);
}

void DeviceStateSet::requireSameDomain(const DeviceStateSet& other) const {
    if (other.state_count_ != state_count_) {
        throw std::invalid_argument("DeviceStateSet: operands range over different state spaces");
    }
}

void DeviceStateSet::clear(cudaStream_t stream) {
    if (words_.empty()) {
        return;
    }
    words_.fillBytes(0, stream);
}

void DeviceStateSet::fillAll(cudaStream_t stream) {
    if (words_.empty()) {
        return;
    }
    const LaunchConfig launch = gridStrideLaunch(wordCount());
    fillKernel<<<launch.grid, launch.block, 0, stream>>>(words_.data(), ~SetWord{0},
                                                         lastWordMaskFor(state_count_),
                                                         wordCount());
    KRIPCUDA_CHECK_LAST_ERROR();
}

void DeviceStateSet::assign(const DeviceStateSet& source, cudaStream_t stream) {
    requireSameDomain(source);
    if (words_.empty()) {
        return;
    }
    const LaunchConfig launch = gridStrideLaunch(wordCount());
    binaryOpKernel<BinaryOp::Assign><<<launch.grid, launch.block, 0, stream>>>(
        words_.data(), source.words_.data(), wordCount());
    KRIPCUDA_CHECK_LAST_ERROR();
}

void DeviceStateSet::unionWith(const DeviceStateSet& other, cudaStream_t stream) {
    requireSameDomain(other);
    if (words_.empty()) {
        return;
    }
    const LaunchConfig launch = gridStrideLaunch(wordCount());
    binaryOpKernel<BinaryOp::Union><<<launch.grid, launch.block, 0, stream>>>(
        words_.data(), other.words_.data(), wordCount());
    KRIPCUDA_CHECK_LAST_ERROR();
}

void DeviceStateSet::intersectWith(const DeviceStateSet& other, cudaStream_t stream) {
    requireSameDomain(other);
    if (words_.empty()) {
        return;
    }
    const LaunchConfig launch = gridStrideLaunch(wordCount());
    binaryOpKernel<BinaryOp::Intersection><<<launch.grid, launch.block, 0, stream>>>(
        words_.data(), other.words_.data(), wordCount());
    KRIPCUDA_CHECK_LAST_ERROR();
}

void DeviceStateSet::subtract(const DeviceStateSet& other, cudaStream_t stream) {
    requireSameDomain(other);
    if (words_.empty()) {
        return;
    }
    const LaunchConfig launch = gridStrideLaunch(wordCount());
    binaryOpKernel<BinaryOp::Difference><<<launch.grid, launch.block, 0, stream>>>(
        words_.data(), other.words_.data(), wordCount());
    KRIPCUDA_CHECK_LAST_ERROR();
}

void DeviceStateSet::complement(cudaStream_t stream) {
    if (words_.empty()) {
        return;
    }
    const LaunchConfig launch = gridStrideLaunch(wordCount());
    complementKernel<<<launch.grid, launch.block, 0, stream>>>(
        words_.data(), lastWordMaskFor(state_count_), wordCount());
    KRIPCUDA_CHECK_LAST_ERROR();
}

std::uint32_t DeviceStateSet::reduceScalar(ScalarReduction reduction, const DeviceStateSet* other,
                                           cudaStream_t stream) const {
    if (words_.empty()) {
        return 0;
    }
    scalar_.fillBytes(0, stream);

    const LaunchConfig launch = gridStrideLaunch(wordCount(), kDefaultBlockSize);
    switch (reduction) {
    case ScalarReduction::Cardinality:
        cardinalityKernel<<<launch.grid, launch.block, 0, stream>>>(words_.data(), wordCount(),
                                                                    scalar_.data());
        break;
    case ScalarReduction::Difference:
        predicateKernel<PredicateOp::Difference><<<launch.grid, launch.block, 0, stream>>>(
            words_.data(), other->words_.data(), wordCount(), scalar_.data());
        break;
    case ScalarReduction::NotSubset:
        predicateKernel<PredicateOp::NotSubset><<<launch.grid, launch.block, 0, stream>>>(
            words_.data(), other->words_.data(), wordCount(), scalar_.data());
        break;
    }
    KRIPCUDA_CHECK_LAST_ERROR();

    std::uint32_t result = 0;
    scalar_.copyToHost(std::span<std::uint32_t>(&result, 1), stream);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

StateId DeviceStateSet::count(cudaStream_t stream) const {
    return reduceScalar(ScalarReduction::Cardinality, nullptr, stream);
}

bool DeviceStateSet::isEmpty(cudaStream_t stream) const { return count(stream) == 0; }

bool DeviceStateSet::equals(const DeviceStateSet& other, cudaStream_t stream) const {
    requireSameDomain(other);
    return reduceScalar(ScalarReduction::Difference, &other, stream) == 0;
}

bool DeviceStateSet::isSubsetOf(const DeviceStateSet& other, cudaStream_t stream) const {
    requireSameDomain(other);
    return reduceScalar(ScalarReduction::NotSubset, &other, stream) == 0;
}

StateSet DeviceStateSet::toHost(cudaStream_t stream) const {
    StateSet host(state_count_);
    if (!words_.empty()) {
        words_.copyToHost(host.words(), stream);
        KRIPCUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return host;
}

void DeviceStateSet::swap(DeviceStateSet& other) noexcept {
    std::swap(words_, other.words_);
    std::swap(state_count_, other.state_count_);
    std::swap(scalar_, other.scalar_);
}

} // namespace kripcuda
