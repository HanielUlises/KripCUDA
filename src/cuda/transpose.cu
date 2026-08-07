#include "kripcuda/cuda/launch.cuh"
#include "kripcuda/cuda/scan.cuh"
#include "kripcuda/cuda/transpose.cuh"

#include <limits>
#include <stdexcept>

namespace kripcuda {
namespace {

/// One thread per edge. The column array is read in order, so the loads are
/// coalesced; the atomics scatter, but an in-degree histogram over |R| edges is
/// bandwidth bound long before it is contention bound for the degree
/// distributions of concurrent models.
__global__ void inDegreeKernel(const StateId* __restrict__ columns, EdgeIndex edgeCount,
                               EdgeIndex* __restrict__ degrees) {
    const auto stride = static_cast<EdgeIndex>(blockDim.x * gridDim.x);
    for (auto edge = static_cast<EdgeIndex>(blockIdx.x * blockDim.x + threadIdx.x);
         edge < edgeCount; edge += stride) {
        atomicAdd(&degrees[columns[edge]], 1U);
    }
}

/// One thread per source state, so that a thread walks a contiguous run of the
/// forward column array and the cursor atomic is taken once per edge rather
/// than once per (edge, lane) pair.
__global__ void scatterKernel(DeviceKripkeView model, EdgeIndex* __restrict__ cursors,
                              StateId* __restrict__ columns) {
    const auto stride = static_cast<StateId>(blockDim.x * gridDim.x);
    for (auto state = static_cast<StateId>(blockIdx.x * blockDim.x + threadIdx.x);
         state < model.stateCount; state += stride) {
        const EdgeIndex end = model.successorEnd(state);
        for (EdgeIndex edge = model.successorBegin(state); edge < end; ++edge) {
            const StateId successor = model.columns[edge];
            columns[atomicAdd(&cursors[successor], 1U)] = state;
        }
    }
}

__global__ void writeTotalKernel(EdgeIndex* __restrict__ rowOffsets, StateId stateCount,
                                 EdgeIndex total) {
    rowOffsets[stateCount] = total;
}

} // namespace

ReverseRelation::ReverseRelation(const DeviceKripke& model, cudaStream_t stream) {
    const StateId states = model.stateCount();
    if (states == 0) {
        return;
    }
    if (model.transitionCount() > std::numeric_limits<EdgeIndex>::max()) {
        throw std::overflow_error("ReverseRelation: relation exceeds the edge index type");
    }
    const auto edges = static_cast<EdgeIndex>(model.transitionCount());

    row_offsets_ = DeviceBuffer<EdgeIndex>(static_cast<std::size_t>(states) + 1);
    columns_ = DeviceBuffer<StateId>(edges);

    DeviceBuffer<EdgeIndex> degrees(states);
    degrees.fillBytes(0, stream);

    const LaunchConfig edgeLaunch = gridStrideLaunch(edges);
    inDegreeKernel<<<edgeLaunch.grid, edgeLaunch.block, 0, stream>>>(model.view().columns, edges,
                                                                     degrees.data());
    KRIPCUDA_CHECK_LAST_ERROR();

    DeviceScan scan;
    // The in-degrees sum to |R| by construction, so the total cannot overflow
    // an edge index here; it is checked above for the forward relation instead.
    const std::uint64_t total =
        scan.exclusiveScan(degrees.data(), row_offsets_.data(), states, stream);

    writeTotalKernel<<<1, 1, 0, stream>>>(row_offsets_.data(), states,
                                          static_cast<EdgeIndex>(total));
    KRIPCUDA_CHECK_LAST_ERROR();

    // The scatter consumes a private copy of the offsets as per-row cursors,
    // leaving the offsets themselves intact.
    DeviceBuffer<EdgeIndex> cursors = row_offsets_.clone(stream);

    const LaunchConfig stateLaunch = gridStrideLaunch(states);
    scatterKernel<<<stateLaunch.grid, stateLaunch.block, 0, stream>>>(model.view(), cursors.data(),
                                                                      columns_.data());
    KRIPCUDA_CHECK_LAST_ERROR();
    KRIPCUDA_CHECK(cudaStreamSynchronize(stream));

    view_.rowOffsets = row_offsets_.data();
    view_.columns = columns_.data();
    view_.stateCount = states;
}

} // namespace kripcuda
