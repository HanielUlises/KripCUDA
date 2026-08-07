/// Builds a two-process mutual exclusion model, explores its reachable state
/// space on the GPU, and verifies three CTL properties over it: safety, the
/// possibility of entering the critical section, and the liveness property that
/// fails for want of a fairness constraint.

#include "kripcuda/cuda/runtime.hpp"
#include "kripcuda/exploration/reachability.hpp"
#include "kripcuda/verification/ctl.hpp"

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

    for (StateId state = 0; state < model.stateCount(); ++state) {
        const StateId pc0 = state / kLocations;
        const StateId pc1 = state % kLocations;
        std::cout << "  (" << locationName(pc0) << ", " << locationName(pc1) << ") level "
                  << reachable.levels[state] << '\n';
    }

    const bool onDevice = hasCudaDevice();
    const auto check = [&](const ctl::Formula& formula) {
        return onDevice ? checkCtlDevice(model, formula) : checkCtlHost(model, formula);
    };

    const ctl::Formula critical0 = ctl::atom(kCritical0);
    const ctl::Formula critical1 = ctl::atom(kCritical1);

    const CtlResult safety = check(ctl::AG(!(critical0 && critical1)));
    const CtlResult possible = check(ctl::AG(ctl::EF(critical0)));
    const CtlResult liveness = check(ctl::AG(ctl::AF(critical0)));

    const auto report = [](const char* text, const CtlResult& result) {
        std::cout << "  " << text << ": " << (result.holdsInAllInitialStates ? "holds" : "fails")
                  << " (" << result.satisfying.count() << " states, " << result.iterations
                  << " iterations)\n";
    };

    std::cout << "properties:\n";
    report("AG !(crit0 && crit1)", safety);
    report("AG EF crit0         ", possible);
    report("AG AF crit0         ", liveness);
    std::cout << "  liveness fails without a fairness constraint: nothing forces\n"
                 "  the scheduler to ever pick a waiting process.\n";

    return safety.holdsInAllInitialStates ? 0 : 1;
}
