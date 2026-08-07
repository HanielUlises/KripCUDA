/// Builds a two-process mutual exclusion model, explores its reachable state
/// space on the GPU, and checks the safety property AG !(crit0 && crit1) by
/// inspecting the reachable states.

#include "kripcuda/cuda/runtime.hpp"
#include "kripcuda/exploration/reachability.hpp"

#include <iostream>

using namespace kripcuda;

namespace {

enum Location : StateId { Idle = 0, Waiting = 1, Critical = 2 };
constexpr StateId kLocations = 3;
constexpr PropositionId kCritical0 = 0;
constexpr PropositionId kCritical1 = 1;

constexpr StateId encode(StateId pc0, StateId pc1) { return pc0 * kLocations + pc1; }

const char* locationName(StateId location) {
    switch (location) {
    case Idle:
        return "idle";
    case Waiting:
        return "wait";
    default:
        return "crit";
    }
}

/// A process leaves its critical section eventually and may only enter it while
/// the other process is outside of its own.
StateId step(StateId location, StateId other) {
    switch (location) {
    case Idle:
        return Waiting;
    case Waiting:
        return other == Critical ? Waiting : Critical;
    default:
        return Idle;
    }
}

KripkeStructure buildModel() {
    KripkeBuilder builder(kLocations * kLocations, 2);
    builder.nameProposition(kCritical0, "crit0").nameProposition(kCritical1, "crit1");
    builder.addInitialState(encode(Idle, Idle));

    for (StateId pc0 = 0; pc0 < kLocations; ++pc0) {
        for (StateId pc1 = 0; pc1 < kLocations; ++pc1) {
            const StateId state = encode(pc0, pc1);
            builder.setLabel(state, kCritical0, pc0 == Critical);
            builder.setLabel(state, kCritical1, pc1 == Critical);
            builder.addTransition(state, encode(step(pc0, pc1), pc1));
            builder.addTransition(state, encode(pc0, step(pc1, pc0)));
        }
    }
    return builder.build();
}

} // namespace

int main() {
    const KripkeStructure model = buildModel();

    std::cout << "states: " << model.stateCount() << ", transitions: " << model.transitionCount()
              << '\n';

    ReachabilityResult reachable;
    if (hasCudaDevice()) {
        std::cout << "exploring on " << cudaDeviceDescription() << '\n';
        reachable = computeReachabilityDevice(model);
    } else {
        std::cout << "no CUDA device visible; exploring on the host\n";
        reachable = computeReachabilityHost(model);
    }

    std::cout << "reachable: " << reachable.reachableCount << ", depth: " << reachable.maxLevel
              << '\n';

    bool safe = true;
    for (StateId state = 0; state < model.stateCount(); ++state) {
        const StateId pc0 = state / kLocations;
        const StateId pc1 = state % kLocations;
        std::cout << "  (" << locationName(pc0) << ", " << locationName(pc1) << ") level "
                  << reachable.levels[state] << '\n';
        if (reachable.isReachable(state) && model.holds(state, kCritical0) &&
            model.holds(state, kCritical1)) {
            safe = false;
        }
    }

    std::cout << "AG !(crit0 && crit1): " << (safe ? "holds" : "violated") << '\n';
    return safe ? 0 : 1;
}
