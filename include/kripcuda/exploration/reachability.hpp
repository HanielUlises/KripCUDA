#pragma once

#include "kripcuda/kripke.hpp"

#include <cstdint>
#include <vector>

namespace kripcuda {

/// Outcome of a breadth-first exploration from the initial states.
struct ReachabilityResult {
    /// BFS distance from the closest initial state, or kUnreachable.
    std::vector<std::int32_t> levels;
    StateId reachableCount = 0;
    std::int32_t maxLevel = kUnreachable;

    [[nodiscard]] bool isReachable(StateId state) const { return levels[state] != kUnreachable; }
};

/// Sequential reference implementation. Used as an oracle for the device path
/// and as a fallback when no CUDA device is available.
[[nodiscard]] ReachabilityResult computeReachabilityHost(const KripkeStructure& structure);

/// Level-synchronous BFS executed on the GPU.
///
/// The frontier order within a level is nondeterministic, but the produced
/// levels are identical to the host implementation: a state's BFS distance does
/// not depend on the order in which its predecessors are expanded.
///
/// Throws kripcuda::CudaError if the CUDA runtime reports a failure.
[[nodiscard]] ReachabilityResult computeReachabilityDevice(const KripkeStructure& structure);

} // namespace kripcuda
