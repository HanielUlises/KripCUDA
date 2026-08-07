#pragma once

#include "kripcuda/cuda/device_buffer.hpp"
#include "kripcuda/kripke.hpp"

namespace kripcuda {

/// Non-owning, trivially copyable view of a Kripke structure in device memory.
/// Passed to kernels by value.
struct DeviceKripkeView {
    const EdgeIndex* rowOffsets = nullptr;
    const StateId* columns = nullptr;
    const LabelWord* labels = nullptr;
    StateId stateCount = 0;
    std::uint32_t labelWordsPerState = 0;

    __device__ EdgeIndex successorBegin(StateId state) const { return rowOffsets[state]; }
    __device__ EdgeIndex successorEnd(StateId state) const { return rowOffsets[state + 1]; }
    __device__ EdgeIndex outDegree(StateId state) const {
        return rowOffsets[state + 1] - rowOffsets[state];
    }

    __device__ bool holds(StateId state, PropositionId proposition) const {
        const LabelWord word =
            labels[static_cast<std::size_t>(state) * labelWordsPerState +
                   proposition / kLabelWordBits];
        return (word >> (proposition % kLabelWordBits)) & LabelWord{1};
    }

    /// Membership test on a successor list, exploiting the CSR invariant that
    /// successor lists are sorted. Used by the product construction, where the
    /// degree of a product state depends on whether a component has a self loop.
    __device__ bool hasTransition(StateId from, StateId to) const {
        EdgeIndex low = rowOffsets[from];
        EdgeIndex high = rowOffsets[from + 1];
        while (low < high) {
            const EdgeIndex middle = low + (high - low) / 2;
            const StateId candidate = columns[middle];
            if (candidate == to) {
                return true;
            }
            if (candidate < to) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return false;
    }
};

/// Structural statistics of a device-resident model, computed on the GPU when
/// the model is created. The explorers use the degree distribution to choose
/// between thread-per-state and warp-per-state expansion, so the cost of this
/// reduction is paid once and amortised over every traversal of the model.
struct ModelStatistics {
    EdgeIndex maxOutDegree = 0;
    double meanOutDegree = 0.0;
};

/// Owns the device-side copy of a Kripke structure.
///
/// A model is uploaded once and then traversed repeatedly; nothing in the
/// exploration or verification layer transfers it back. Models built on the
/// device — products, for instance — are never on the host at all.
class DeviceKripke {
public:
    explicit DeviceKripke(const KripkeStructure& structure, cudaStream_t stream = nullptr);

    /// Adopts device buffers that already satisfy the CSR invariants: sorted,
    /// duplicate-free successor lists and a total transition relation. The
    /// device-side constructions in this library guarantee them by
    /// construction, which is why they bypass KripkeBuilder.
    DeviceKripke(DeviceBuffer<EdgeIndex> rowOffsets, DeviceBuffer<StateId> columns,
                 DeviceBuffer<LabelWord> labels, std::vector<StateId> initialStates,
                 StateId stateCount, std::uint32_t propositionCount, cudaStream_t stream = nullptr);

    [[nodiscard]] DeviceKripkeView view() const noexcept { return view_; }
    [[nodiscard]] StateId stateCount() const noexcept { return view_.stateCount; }
    [[nodiscard]] std::size_t transitionCount() const noexcept { return columns_.size(); }
    [[nodiscard]] std::uint32_t propositionCount() const noexcept { return proposition_count_; }
    [[nodiscard]] const ModelStatistics& statistics() const noexcept { return statistics_; }

    [[nodiscard]] std::span<const StateId> initialStates() const noexcept {
        return initial_states_host_;
    }
    [[nodiscard]] const DeviceBuffer<StateId>& initialStatesDevice() const noexcept {
        return initial_states_;
    }

    /// Device-to-device copy of the whole model.
    [[nodiscard]] DeviceKripke clone(cudaStream_t stream = nullptr) const;

    /// Copies the model back to the host. Intended for inspection, persistence
    /// and testing; the exploration path never needs it.
    [[nodiscard]] KripkeStructure download(cudaStream_t stream = nullptr) const;

private:
    void computeStatistics(cudaStream_t stream);

    DeviceBuffer<EdgeIndex> row_offsets_;
    DeviceBuffer<StateId> columns_;
    DeviceBuffer<LabelWord> labels_;
    DeviceBuffer<StateId> initial_states_;
    std::vector<StateId> initial_states_host_;
    std::uint32_t proposition_count_ = 0;
    ModelStatistics statistics_;
    DeviceKripkeView view_;
};

} // namespace kripcuda
