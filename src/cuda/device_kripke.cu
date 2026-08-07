#include "kripcuda/cuda/device_kripke.hpp"
#include "kripcuda/cuda/runtime.hpp"

namespace kripcuda {

int cudaDeviceCount() noexcept {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        // Clear the sticky error so later checked calls report their own status.
        static_cast<void>(cudaGetLastError());
        return 0;
    }
    return count;
}

std::string cudaDeviceDescription(int device) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return {};
    }
    return std::string(properties.name) + " (sm_" + std::to_string(properties.major) +
           std::to_string(properties.minor) + ')';
}

DeviceKripke::DeviceKripke(const KripkeStructure& structure, cudaStream_t stream)
    : row_offsets_(structure.rowOffsets().size()),
      columns_(structure.columns().size()),
      labels_(structure.labels().size()),
      initial_states_(structure.initialStates().size()),
      initial_states_host_(structure.initialStates().begin(), structure.initialStates().end()) {
    row_offsets_.copyFromHost(structure.rowOffsets(), stream);
    columns_.copyFromHost(structure.columns(), stream);
    if (!labels_.empty()) {
        labels_.copyFromHost(structure.labels(), stream);
    }
    initial_states_.copyFromHost(structure.initialStates(), stream);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

    view_.rowOffsets = row_offsets_.data();
    view_.columns = columns_.data();
    view_.labels = labels_.data();
    view_.stateCount = structure.stateCount();
    view_.labelWordsPerState = structure.labelWordsPerState();
}

} // namespace kripcuda
