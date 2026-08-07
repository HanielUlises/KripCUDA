/// Builds compositions of an increasing number of concurrent processes
/// entirely on the GPU, explores each reachable state space, and verifies
/// mutual exclusion of every component on the composed model.
///
/// Nothing but the single-process component ever crosses the bus: each
/// composition is a product of device-resident models, and the explorer and the
/// CTL evaluator both consume it in place.

#include "kripcuda/cuda/product.cuh"
#include "kripcuda/cuda/runtime.hpp"
#include "kripcuda/cuda/stream.cuh"
#include "kripcuda/exploration/reachability_device.cuh"
#include "kripcuda/verification/ctl_device.cuh"

#include <cstdio>

using namespace kripcuda;

namespace {

enum Location : StateId { Idle = 0, Waiting = 1, Critical = 2 };
constexpr StateId kLocations = 3;

constexpr StateId encode(StateId first, StateId second) { return first * kLocations + second; }

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

/// One two-process mutual exclusion component, the building block composed
/// below. Propositions 0 and 1 mark each process being in its critical section.
KripkeStructure buildComponent() {
    KripkeBuilder builder(kLocations * kLocations, 2);
    builder.nameProposition(0, "crit0").nameProposition(1, "crit1");
    builder.addInitialState(encode(Idle, Idle));

    for (StateId first = 0; first < kLocations; ++first) {
        for (StateId second = 0; second < kLocations; ++second) {
            const StateId state = encode(first, second);
            builder.setLabel(state, 0, first == Critical);
            builder.setLabel(state, 1, second == Critical);
            builder.addTransition(state, encode(step(first, second), second));
            builder.addTransition(state, encode(first, step(second, first)));
        }
    }
    return builder.build();
}

/// Mutual exclusion of every component of the composition: a conjunction of one
/// AG per component, sharing subformulas through the formula DAG.
ctl::Formula safetyOfEveryComponent(unsigned components) {
    ctl::Formula safety = ctl::constantTrue();
    for (unsigned component = 0; component < components; ++component) {
        safety = safety && ctl::AG(!(ctl::atom(component * 2) && ctl::atom(component * 2 + 1)));
    }
    return safety;
}

} // namespace

int main() {
    if (!hasCudaDevice()) {
        std::puts("no CUDA device visible");
        return 1;
    }
    std::printf("device: %s\n\n", cudaDeviceDescription().c_str());

    Stream stream;
    const DeviceKripke component(buildComponent(), stream);

    std::puts("procs      states    transitions   reachable  depth   explore    verify   safe");
    std::puts("--------------------------------------------------------------------------------");

    for (unsigned components = 1; components <= 6; ++components) {
        const DeviceKripke model =
            buildPower(component, components, ProductKind::Interleaving, stream);

        float exploreMilliseconds = 0.0F;
        ReachabilityResult reachable;
        {
            const ScopedTimer timer(exploreMilliseconds, stream);
            reachable = computeReachabilityDevice(model, stream);
        }

        float verifyMilliseconds = 0.0F;
        CtlResult safety;
        {
            const ScopedTimer timer(verifyMilliseconds, stream);
            DeviceCtlEvaluator evaluator(model, stream);
            safety = evaluator.check(safetyOfEveryComponent(components));
        }

        std::printf("%5u %11u %14zu %11u %6d %8.2fms %8.2fms   %s\n", 2 * components,
                    model.stateCount(), model.transitionCount(), reachable.reachableCount,
                    reachable.maxLevel, static_cast<double>(exploreMilliseconds),
                    static_cast<double>(verifyMilliseconds),
                    safety.holdsInAllInitialStates ? "yes" : "NO");
    }

    return 0;
}
