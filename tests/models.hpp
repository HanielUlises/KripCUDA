#pragma once

#include "kripcuda/kripke.hpp"

#include <random>

namespace kripcuda::testing {

/// Deterministic pseudo-random model with a total transition relation.
/// `reachableFraction` bounds the successors drawn from the low state indices,
/// which leaves the remaining states unreachable from state 0 and exercises the
/// unreachable-state paths of the explorers.
inline KripkeStructure randomModel(StateId stateCount, std::uint32_t outDegree,
                                   std::uint32_t propositionCount, std::uint64_t seed,
                                   double reachableFraction = 1.0) {
    std::mt19937_64 engine(seed);
    const auto reachableStates =
        static_cast<StateId>(static_cast<double>(stateCount) * reachableFraction);
    std::uniform_int_distribution<StateId> reachableTarget(0, (reachableStates ? reachableStates : 1) - 1);
    std::uniform_int_distribution<StateId> anyTarget(0, stateCount - 1);
    std::bernoulli_distribution labelBit(0.3);

    KripkeBuilder builder(stateCount, propositionCount);
    builder.addInitialState(0);
    for (StateId state = 0; state < stateCount; ++state) {
        const bool reachablePart = state < reachableStates;
        for (std::uint32_t edge = 0; edge < outDegree; ++edge) {
            builder.addTransition(state, reachablePart ? reachableTarget(engine) : anyTarget(engine));
        }
        for (PropositionId proposition = 0; proposition < propositionCount; ++proposition) {
            if (labelBit(engine)) {
                builder.setLabel(state, proposition);
            }
        }
    }
    return builder.build();
}

/// Two-process mutual exclusion model used as a small, human-checkable fixture.
/// A state encodes (pc0, pc1) with pc in {idle, waiting, critical}; the process
/// scheduler is nondeterministic and a process may only enter its critical
/// section when the other one is not in it.
inline KripkeStructure mutualExclusionModel() {
    constexpr StateId kLocations = 3;
    constexpr StateId kStates = kLocations * kLocations;
    constexpr PropositionId kCritical0 = 0;
    constexpr PropositionId kCritical1 = 1;

    enum Location : StateId { Idle = 0, Waiting = 1, Critical = 2 };
    const auto encode = [](StateId pc0, StateId pc1) { return pc0 * kLocations + pc1; };

    KripkeBuilder builder(kStates, 2);
    builder.nameProposition(kCritical0, "crit0").nameProposition(kCritical1, "crit1");
    builder.addInitialState(encode(Idle, Idle));

    for (StateId pc0 = 0; pc0 < kLocations; ++pc0) {
        for (StateId pc1 = 0; pc1 < kLocations; ++pc1) {
            const StateId state = encode(pc0, pc1);
            if (pc0 == Critical) {
                builder.setLabel(state, kCritical0);
            }
            if (pc1 == Critical) {
                builder.setLabel(state, kCritical1);
            }

            const auto step = [&](StateId location, StateId other) -> StateId {
                switch (location) {
                case Idle:
                    return Waiting;
                case Waiting:
                    return other == Critical ? Waiting : Critical;
                default:
                    return Idle;
                }
            };
            builder.addTransition(state, encode(step(pc0, pc1), pc1));
            builder.addTransition(state, encode(pc0, step(pc1, pc0)));
        }
    }
    return builder.build();
}

} // namespace kripcuda::testing
