#pragma once

#include "kripcuda/cuda/device_buffer.hpp"
#include "kripcuda/state_set.hpp"

namespace kripcuda {

/// Trivially copyable view of a device-resident state set, passed to kernels.
struct DeviceStateSetView {
    SetWord* words = nullptr;
    StateId stateCount = 0;
    std::uint32_t wordCount = 0;

    __device__ bool contains(StateId state) const {
        return (words[state / kSetWordBits] >> (state % kSetWordBits) & SetWord{1}) != 0;
    }
};

struct ConstDeviceStateSetView {
    const SetWord* words = nullptr;
    StateId stateCount = 0;
    std::uint32_t wordCount = 0;

    __device__ bool contains(StateId state) const {
        return (words[state / kSetWordBits] >> (state % kSetWordBits) & SetWord{1}) != 0;
    }

    /// Membership test that tolerates a null set, read as the empty set. Used
    /// by the fixpoint kernels, where an absent operand is the neutral element.
    __device__ static bool containsOrEmpty(const SetWord* setWords, StateId state) {
        return setWords != nullptr &&
               (setWords[state / kSetWordBits] >> (state % kSetWordBits) & SetWord{1}) != 0;
    }
};

/// Device-resident characteristic function of a subset of the state space,
/// with the full boolean algebra implemented as elementwise kernels over
/// 32-bit words.
///
/// The padding bits above stateCount in the last word are held at zero, exactly
/// as in the host StateSet: complement masks them off and every producing
/// kernel derives them from an out-of-range predicate that is false. Without
/// that invariant the reductions below would count states that do not exist.
class DeviceStateSet {
public:
    DeviceStateSet() = default;
    explicit DeviceStateSet(StateId stateCount);
    DeviceStateSet(const StateSet& host, cudaStream_t stream = nullptr);

    [[nodiscard]] StateId stateCount() const noexcept { return state_count_; }
    [[nodiscard]] std::uint32_t wordCount() const noexcept {
        return static_cast<std::uint32_t>(words_.size());
    }
    [[nodiscard]] SetWord* words() noexcept { return words_.data(); }
    [[nodiscard]] const SetWord* words() const noexcept { return words_.data(); }

    [[nodiscard]] DeviceStateSetView view() noexcept {
        return DeviceStateSetView{words_.data(), state_count_, wordCount()};
    }
    [[nodiscard]] ConstDeviceStateSetView view() const noexcept {
        return ConstDeviceStateSetView{words_.data(), state_count_, wordCount()};
    }

    void clear(cudaStream_t stream = nullptr);
    void fillAll(cudaStream_t stream = nullptr);
    void assign(const DeviceStateSet& source, cudaStream_t stream = nullptr);

    void unionWith(const DeviceStateSet& other, cudaStream_t stream = nullptr);
    void intersectWith(const DeviceStateSet& other, cudaStream_t stream = nullptr);
    void subtract(const DeviceStateSet& other, cudaStream_t stream = nullptr);
    void complement(cudaStream_t stream = nullptr);

    /// Cardinality, by a population count reduced across the grid.
    [[nodiscard]] StateId count(cudaStream_t stream = nullptr) const;
    [[nodiscard]] bool isEmpty(cudaStream_t stream = nullptr) const;
    [[nodiscard]] bool equals(const DeviceStateSet& other, cudaStream_t stream = nullptr) const;
    [[nodiscard]] bool isSubsetOf(const DeviceStateSet& other, cudaStream_t stream = nullptr) const;

    [[nodiscard]] StateSet toHost(cudaStream_t stream = nullptr) const;

    void swap(DeviceStateSet& other) noexcept;

private:
    /// Reductions that produce a single scalar and therefore a single small
    /// device-to-host transfer.
    enum class ScalarReduction {
        Cardinality,   ///< number of set bits
        Difference,    ///< 1 if the two sets differ
        NotSubset,     ///< 1 if some element is missing from the other set
    };

    void requireSameDomain(const DeviceStateSet& other) const;
    [[nodiscard]] std::uint32_t reduceScalar(ScalarReduction reduction, const DeviceStateSet* other,
                                             cudaStream_t stream) const;

    DeviceBuffer<SetWord> words_;
    StateId state_count_ = 0;
    mutable DeviceBuffer<std::uint32_t> scalar_;
};

} // namespace kripcuda
