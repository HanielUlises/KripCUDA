#pragma once

#include "kripcuda/cuda/device_kripke.cuh"
#include "kripcuda/cuda/device_state_set.cuh"
#include "kripcuda/cuda/transpose.cuh"

namespace kripcuda {

/// Trivially copyable view of an SCC partition, passed to kernels by value.
struct SccPartitionView {
    /// Representative of the component of a state, or kInvalidState if the
    /// state lies outside the region the decomposition was computed over.
    const StateId* component = nullptr;
    /// Number of states in the component of a representative. Only the entries
    /// indexed by a representative are meaningful.
    const StateId* size = nullptr;
    StateId stateCount = 0;

    __device__ bool assigned(StateId state) const { return component[state] != kInvalidState; }

    /// A component is nontrivial — that is, its states lie on a cycle — when it
    /// holds more than one state, or exactly one that carries a self loop.
    __device__ bool onCycle(DeviceKripkeView model, StateId state) const {
        const StateId representative = component[state];
        if (representative == kInvalidState) {
            return false;
        }
        return size[representative] > 1 || model.hasTransition(state, state);
    }
};

struct SccStatistics {
    StateId componentCount = 0;
    /// Components whose states lie on a cycle.
    StateId nontrivialCount = 0;
    StateId largestComponent = 0;
    /// Colouring rounds the decomposition took; a diagnostic, since the round
    /// count is what separates an easy state space from an adversarial one.
    std::uint32_t rounds = 0;
};

/// Strongly connected components of a device-resident model, computed and kept
/// on the GPU.
///
/// The partition is stored as one representative per state rather than as a
/// dense component numbering: the representative is the largest state index in
/// the component, which the colouring algorithm produces directly and which
/// needs no renumbering pass to be usable as a key.
class SccPartition {
public:
    SccPartition() = default;
    SccPartition(DeviceBuffer<StateId> component, DeviceBuffer<StateId> size, StateId stateCount,
                 SccStatistics statistics);

    [[nodiscard]] StateId stateCount() const noexcept { return state_count_; }
    [[nodiscard]] const SccStatistics& statistics() const noexcept { return statistics_; }

    [[nodiscard]] SccPartitionView view() const noexcept {
        return SccPartitionView{component_.data(), size_.data(), state_count_};
    }

    /// Component representative of every state, for host-side inspection and
    /// testing; the checking path never needs it.
    [[nodiscard]] std::vector<StateId> componentsToHost(cudaStream_t stream = nullptr) const {
        return component_.toHost(stream);
    }

    /// The states that lie on a cycle: the union of the nontrivial components.
    [[nodiscard]] DeviceStateSet cyclicStates(const DeviceKripke& model,
                                              cudaStream_t stream = nullptr) const;

private:
    DeviceBuffer<StateId> component_;
    DeviceBuffer<StateId> size_;
    StateId state_count_ = 0;
    SccStatistics statistics_;
};

/// Decomposes the subgraph of `model` induced by `region` into its strongly
/// connected components. A null region denotes the whole state space.
///
/// The algorithm is the parallel colouring decomposition: every live state is
/// coloured with the largest state index that can reach it, which makes each
/// colour class a union of an SCC and the part of its forward set that no
/// larger state reaches; the class of a colour that is its own colour is then
/// closed backwards to yield one complete SCC per round. Rounds are preceded by
/// trimming, which removes the states with no live successor or no live
/// predecessor — in a concurrent model that is most of them, and each one
/// removed is a component the colouring never has to look at.
///
/// Nothing crosses the bus except one flag per kernel iteration: the model, the
/// partition and the region all stay on the device.
[[nodiscard]] SccPartition computeSccDevice(const DeviceKripke& model,
                                            const ReverseRelation& reverse,
                                            const DeviceStateSet* region = nullptr,
                                            cudaStream_t stream = nullptr);

/// Convenience overload that builds the transpose it needs. Prefer the overload
/// above when several decompositions are taken over the same model, since the
/// transpose does not depend on the region.
[[nodiscard]] SccPartition computeSccDevice(const DeviceKripke& model,
                                            cudaStream_t stream = nullptr);

} // namespace kripcuda
