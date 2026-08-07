#pragma once

#include "kripcuda/cuda/device_kripke.cuh"

namespace kripcuda {

/// Non-owning, trivially copyable view of the reverse transition relation
/// R⁻¹ = { (s', s) | R(s, s') } in CSR form. Passed to kernels by value.
struct ReverseRelationView {
    const EdgeIndex* rowOffsets = nullptr;
    const StateId* columns = nullptr;
    StateId stateCount = 0;

    __device__ EdgeIndex predecessorBegin(StateId state) const { return rowOffsets[state]; }
    __device__ EdgeIndex predecessorEnd(StateId state) const { return rowOffsets[state + 1]; }
    __device__ EdgeIndex inDegree(StateId state) const {
        return rowOffsets[state + 1] - rowOffsets[state];
    }
};

/// The transposed transition relation of a device-resident model.
///
/// Most of the library deliberately avoids the transpose: the CTL fixpoints
/// compute pre∃ as a gather over the forward relation, so they never need it.
/// The SCC decomposition does, because trimming asks whether a state has any
/// live predecessor — a question the forward relation can only answer by
/// scanning the whole edge list.
///
/// The transpose is built entirely on the device: in-degrees are counted with
/// one atomic per edge, turned into row offsets by a device-wide exclusive
/// scan, and filled by a second pass over the edges. The order of the states
/// within a predecessor list is unspecified — the scatter races on a per-row
/// cursor — which is why this is a ReverseRelation and not a KripkeStructure:
/// nothing here relies on the sortedness invariant of the model layer.
class ReverseRelation {
public:
    explicit ReverseRelation(const DeviceKripke& model, cudaStream_t stream = nullptr);

    [[nodiscard]] ReverseRelationView view() const noexcept { return view_; }
    [[nodiscard]] StateId stateCount() const noexcept { return view_.stateCount; }
    [[nodiscard]] std::size_t transitionCount() const noexcept { return columns_.size(); }

private:
    DeviceBuffer<EdgeIndex> row_offsets_;
    DeviceBuffer<StateId> columns_;
    ReverseRelationView view_;
};

} // namespace kripcuda
