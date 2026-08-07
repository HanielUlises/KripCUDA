#include "kripcuda/cuda/device_kripke.hpp"
#include "kripcuda/cuda/launch.hpp"
#include "kripcuda/cuda/reduction.hpp"
#include "kripcuda/cuda/runtime.hpp"

#include <stdexcept>

namespace kripcuda {
namespace {

__global__ void maxDegreeKernel(const EdgeIndex* __restrict__ rowOffsets, StateId stateCount,
                                EdgeIndex* __restrict__ result) {
    __shared__ EdgeIndex shared[kDefaultBlockSize / kWarpSize];

    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    EdgeIndex local = 0;
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < stateCount; state += stride) {
        const EdgeIndex degree = rowOffsets[state + 1] - rowOffsets[state];
        local = degree > local ? degree : local;
    }

    const EdgeIndex blockMax = blockReduceMax(local, shared, EdgeIndex{0});
    if (threadIdx.x == 0 && blockMax != 0) {
        atomicMax(result, blockMax);
    }
}

} // namespace

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
      initial_states_host_(structure.initialStates().begin(), structure.initialStates().end()),
      proposition_count_(structure.propositionCount()) {
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

    computeStatistics(stream);
}

DeviceKripke::DeviceKripke(DeviceBuffer<EdgeIndex> rowOffsets, DeviceBuffer<StateId> columns,
                           DeviceBuffer<LabelWord> labels, std::vector<StateId> initialStates,
                           StateId stateCount, std::uint32_t propositionCount, cudaStream_t stream)
    : row_offsets_(std::move(rowOffsets)),
      columns_(std::move(columns)),
      labels_(std::move(labels)),
      initial_states_(initialStates.size()),
      initial_states_host_(std::move(initialStates)),
      proposition_count_(propositionCount) {
    if (row_offsets_.size() != static_cast<std::size_t>(stateCount) + 1) {
        throw std::invalid_argument("DeviceKripke: row offsets do not match the state count");
    }
    initial_states_.copyFromHost(std::span<const StateId>(initial_states_host_), stream);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

    view_.rowOffsets = row_offsets_.data();
    view_.columns = columns_.data();
    view_.labels = labels_.data();
    view_.stateCount = stateCount;
    view_.labelWordsPerState = labelWordsFor(propositionCount);

    computeStatistics(stream);
}

void DeviceKripke::computeStatistics(cudaStream_t stream) {
    statistics_.meanOutDegree =
        view_.stateCount == 0
            ? 0.0
            : static_cast<double>(columns_.size()) / static_cast<double>(view_.stateCount);

    if (view_.stateCount == 0) {
        return;
    }

    DeviceBuffer<EdgeIndex> result(1);
    result.fillBytes(0, stream);

    const LaunchConfig launch = gridStrideLaunch(view_.stateCount);
    maxDegreeKernel<<<launch.grid, launch.block, 0, stream>>>(view_.rowOffsets, view_.stateCount,
                                                              result.data());
    KRIPCUDA_CHECK_LAST_ERROR();

    result.copyToHost(std::span<EdgeIndex>(&statistics_.maxOutDegree, 1), stream);
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));
}

DeviceKripke DeviceKripke::clone(cudaStream_t stream) const {
    return DeviceKripke(row_offsets_.clone(stream), columns_.clone(stream), labels_.clone(stream),
                        initial_states_host_, view_.stateCount, proposition_count_, stream);
}

KripkeStructure DeviceKripke::download(cudaStream_t stream) const {
    std::vector<EdgeIndex> rowOffsets = row_offsets_.toHost(stream);
    std::vector<StateId> columns = columns_.toHost(stream);
    std::vector<LabelWord> labels = labels_.toHost(stream);
    return KripkeStructure::fromValidatedCsr(std::move(rowOffsets), std::move(columns),
                                             initial_states_host_, std::move(labels),
                                             proposition_count_);
}

} // namespace kripcuda
