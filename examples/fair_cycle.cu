/// Fair-cycle detection on the GPU.
///
/// A liveness property of a concurrent model is only meaningful under a
/// fairness assumption: without one, the scheduler is free to starve a process
/// for ever and every "eventually" fails for uninteresting reasons. This
/// example builds a composition of mutual exclusion processes on the device,
/// decomposes it into strongly connected components there, and asks whether a
/// fair execution exists — first under the assumption that every process enters
/// its critical section infinitely often, then under an assumption no execution
/// of the model can satisfy.
///
/// Nothing but the single-process component crosses the bus: the composition,
/// the transpose, the SCC decomposition and the fairness check all run on
/// device-resident data.

#include "kripcuda/cuda/product.cuh"
#include "kripcuda/cuda/runtime.hpp"
#include "kripcuda/cuda/stream.cuh"
#include "kripcuda/cuda/transpose.cuh"
#include "kripcuda/verification/fairness_device.cuh"

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

/// One two-process mutual exclusion component. Propositions 0 and 1 mark each
/// process being in its critical section; composing k copies gives 2k of them.
KripkeStructure buildComponent() {
    KripkeBuilder builder(kLocations * kLocations, 2);
    builder.nameProposition(0, "crit0").nameProposition(1, "crit1");
    builder.addInitialState(encode(Idle, Idle));

    for (StateId first = 0; first < kLocations; ++first) {
        for (StateId second = 0; second < kLocations; ++second) {
            const StateId state = encode(first, second);
            if (first == Critical) {
                builder.setLabel(state, 0);
            }
            if (second == Critical) {
                builder.setLabel(state, 1);
            }
            builder.addTransition(state, encode(step(first, second), second));
            builder.addTransition(state, encode(first, step(second, first)));
        }
    }
    return builder.build();
}

/// The set of states satisfying an atomic proposition, built on the device by
/// the CTL evaluator's atom kernel by way of a one-node formula.
DeviceStateSet statesSatisfying(const DeviceKripke& model, PropositionId proposition,
                                cudaStream_t stream) {
    StateSet host(model.stateCount());
    const KripkeStructure downloaded = model.download(stream);
    for (StateId state = 0; state < downloaded.stateCount(); ++state) {
        if (downloaded.holds(state, proposition)) {
            host.insert(state);
        }
    }
    return DeviceStateSet(host, stream);
}

void report(const char* what, const FairCycleResult& result, StateId states) {
    std::printf("  %-34s %-3s  witnesses %8u / %-8u  components %8u (%u cyclic, largest %u)\n",
                what, result.holds ? "yes" : "no", result.witnessCount, states,
                result.scc.componentCount, result.scc.nontrivialCount,
                result.scc.largestComponent);
}

} // namespace

int main() {
    if (!hasCudaDevice()) {
        std::printf("no CUDA device visible; nothing to do\n");
        return 0;
    }
    std::printf("CUDA device: %s\n\n", cudaDeviceDescription().c_str());

    Stream stream;
    const DeviceKripke component(buildComponent(), stream);

    for (unsigned copies = 1; copies <= 5; ++copies) {
        const DeviceKripke composed =
            buildPower(component, copies, ProductKind::Interleaving, stream);

        std::printf("%u component(s): %u states, %zu transitions\n", copies, composed.stateCount(),
                    composed.transitionCount());

        // Every process enters its critical section infinitely often.
        FairnessCondition fairness;
        for (std::uint32_t proposition = 0; proposition < composed.propositionCount();
             ++proposition) {
            fairness.add(statesSatisfying(composed, proposition, stream));
        }
        report("all processes progress:", checkFairCycle(composed, fairness, stream),
               composed.stateCount());

        // No fairness assumption at all: every cycle is fair, so this is plain
        // cycle detection and it succeeds in any model with a total relation.
        report("any infinite execution:", checkFairCycle(composed, FairnessCondition(), stream),
               composed.stateCount());

        // An assumption no execution satisfies, as a control.
        FairnessCondition impossible;
        impossible.add(DeviceStateSet(StateSet(composed.stateCount()), stream));
        report("unsatisfiable assumption:", checkFairCycle(composed, impossible, stream),
               composed.stateCount());
    }

    return 0;
}
