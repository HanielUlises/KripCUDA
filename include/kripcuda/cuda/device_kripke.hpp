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

    __device__ bool holds(StateId state, PropositionId proposition) const {
        const LabelWord word =
            labels[static_cast<std::size_t>(state) * labelWordsPerState +
                   proposition / kLabelWordBits];
        return (word >> (proposition % kLabelWordBits)) & LabelWord{1};
    }
};

/// Owns the device-side copy of a Kripke structure. Uploading is the only
/// host-to-device transfer required by the exploration kernels, so a single
/// instance should be reused across verification runs on the same model.
class DeviceKripke {
public:
    explicit DeviceKripke(const KripkeStructure& structure, cudaStream_t stream = nullptr);

    [[nodiscard]] DeviceKripkeView view() const noexcept { return view_; }
    [[nodiscard]] StateId stateCount() const noexcept { return view_.stateCount; }
    [[nodiscard]] std::span<const StateId> initialStates() const noexcept {
        return initial_states_host_;
    }
    [[nodiscard]] const DeviceBuffer<StateId>& initialStatesDevice() const noexcept {
        return initial_states_;
    }

private:
    DeviceBuffer<EdgeIndex> row_offsets_;
    DeviceBuffer<StateId> columns_;
    DeviceBuffer<LabelWord> labels_;
    DeviceBuffer<StateId> initial_states_;
    std::vector<StateId> initial_states_host_;
    DeviceKripkeView view_;
};

} // namespace kripcuda
