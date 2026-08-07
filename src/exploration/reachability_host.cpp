#include "kripcuda/exploration/reachability.hpp"

#include <algorithm>

namespace kripcuda {

ReachabilityResult computeReachabilityHost(const KripkeStructure& structure) {
    const StateId states = structure.stateCount();

    ReachabilityResult result;
    result.levels.assign(states, kUnreachable);

    // A vector used as a FIFO with an explicit read cursor: the BFS visits each
    // state at most once, so the queue never exceeds |S| entries and no
    // reallocation-heavy container is needed.
    std::vector<StateId> queue;
    queue.reserve(states);

    for (const StateId initial : structure.initialStates()) {
        if (result.levels[initial] == kUnreachable) {
            result.levels[initial] = 0;
            queue.push_back(initial);
        }
    }

    for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
        const StateId state = queue[cursor];
        const std::int32_t nextLevel = result.levels[state] + 1;
        for (const StateId successor : structure.successors(state)) {
            if (result.levels[successor] == kUnreachable) {
                result.levels[successor] = nextLevel;
                queue.push_back(successor);
            }
        }
    }

    result.reachableCount = static_cast<StateId>(queue.size());
    for (const std::int32_t level : result.levels) {
        result.maxLevel = std::max(result.maxLevel, level);
    }
    return result;
}

} // namespace kripcuda
