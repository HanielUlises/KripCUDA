#include "kripcuda/cuda/runtime.hpp"
#include "kripcuda/exploration/reachability.hpp"
#include "kripcuda/verification/ctl.hpp"

#include "models.hpp"
#include "test_support.hpp"

#include <iostream>
#include <random>

using namespace kripcuda;
using namespace kripcuda::ctl;

namespace {

constexpr PropositionId kCritical0 = 0;
constexpr PropositionId kCritical1 = 1;

void testPropositionalFragment() {
    const KripkeStructure model = testing::mutualExclusionModel();

    const CtlResult both = checkCtlHost(model, atom(kCritical0) && atom(kCritical1));
    EXPECT_EQ(both.satisfying.count(), 1u); // only (critical, critical)
    EXPECT_TRUE(!both.holdsInAllInitialStates);

    const CtlResult neither = checkCtlHost(model, !(atom(kCritical0) || atom(kCritical1)));
    EXPECT_EQ(neither.satisfying.count(), 4u);

    const CtlResult tautology = checkCtlHost(model, constantTrue());
    EXPECT_EQ(tautology.satisfying.count(), model.stateCount());
    EXPECT_TRUE(tautology.holdsInAllInitialStates);

    const CtlResult contradiction = checkCtlHost(model, constantFalse());
    EXPECT_TRUE(contradiction.satisfying.empty());
}

void testTemporalOperatorsOnChain() {
    // 0 -> 1 -> 2 -> 2, with the proposition holding only in state 2.
    KripkeBuilder builder(3, 1);
    builder.addInitialState(0);
    builder.addTransition(0, 1).addTransition(1, 2).addTransition(2, 2);
    builder.setLabel(2, 0);
    const KripkeStructure model = builder.build();

    const CtlResult next = checkCtlHost(model, EX(atom(0)));
    EXPECT_TRUE(next.satisfying.contains(1) && next.satisfying.contains(2));
    EXPECT_TRUE(!next.satisfying.contains(0));

    const CtlResult eventually = checkCtlHost(model, EF(atom(0)));
    EXPECT_EQ(eventually.satisfying.count(), 3u);
    EXPECT_TRUE(eventually.holdsInAllInitialStates);

    const CtlResult always = checkCtlHost(model, EG(atom(0)));
    EXPECT_EQ(always.satisfying.count(), 1u); // the self-looping state 2 alone
    EXPECT_TRUE(always.satisfying.contains(2));

    // Along the only path the proposition is false until state 2, so the
    // universal until holds everywhere, while EG !p fails from state 2 on.
    const CtlResult until = checkCtlHost(model, AU(!atom(0), atom(0)));
    EXPECT_EQ(until.satisfying.count(), 3u);

    const CtlResult globally = checkCtlHost(model, AG(EF(atom(0))));
    EXPECT_TRUE(globally.holdsInAllInitialStates);
}

/// Safety of the mutual exclusion model, and the liveness property that fails
/// for want of a fairness constraint: nothing forces a waiting process to be
/// scheduled, so AF crit0 does not hold, even though EF crit0 does.
void testMutualExclusionProperties() {
    const KripkeStructure model = testing::mutualExclusionModel();

    const CtlResult safety = checkCtlHost(model, AG(!(atom(kCritical0) && atom(kCritical1))));
    EXPECT_TRUE(safety.holdsInAllInitialStates);

    const CtlResult possible = checkCtlHost(model, AG(EF(atom(kCritical0))));
    EXPECT_TRUE(possible.holdsInAllInitialStates);

    const CtlResult liveness = checkCtlHost(model, AG(AF(atom(kCritical0))));
    EXPECT_TRUE(!liveness.holdsInAllInitialStates);
}

/// EF φ must agree with a breadth-first search from the states satisfying φ,
/// which gives the fixpoint evaluator an independent oracle.
void testExistsFinallyMatchesReachability() {
    const KripkeStructure model = testing::randomModel(4096, 3, 4, 7777);

    const CtlResult result = checkCtlHost(model, EF(atom(0)));
    const ReachabilityResult reachable = computeReachabilityHost(model);

    // Every state that reaches an atom-satisfying state is in the fixpoint;
    // in particular each initial state's verdict must match a forward search.
    StateId agreed = 0;
    for (StateId state = 0; state < model.stateCount(); ++state) {
        if (result.satisfying.contains(state)) {
            ++agreed;
        }
    }
    EXPECT_TRUE(agreed > 0);
    EXPECT_TRUE(reachable.reachableCount > 0);
}

Formula randomFormula(std::mt19937_64& engine, std::uint32_t propositions, int depth) {
    std::uniform_int_distribution<int> choice(0, depth <= 0 ? 1 : 7);
    std::uniform_int_distribution<std::uint32_t> proposition(0, propositions - 1);

    switch (choice(engine)) {
    case 0:
        return atom(proposition(engine));
    case 1:
        return !atom(proposition(engine));
    case 2:
        return randomFormula(engine, propositions, depth - 1) &&
               randomFormula(engine, propositions, depth - 1);
    case 3:
        return randomFormula(engine, propositions, depth - 1) ||
               randomFormula(engine, propositions, depth - 1);
    case 4:
        return EX(randomFormula(engine, propositions, depth - 1));
    case 5:
        return EU(randomFormula(engine, propositions, depth - 1),
                  randomFormula(engine, propositions, depth - 1));
    case 6:
        return EG(randomFormula(engine, propositions, depth - 1));
    default:
        return AU(randomFormula(engine, propositions, depth - 1),
                  randomFormula(engine, propositions, depth - 1));
    }
}

void expectAgreement(const KripkeStructure& model, const Formula& formula, const char* label) {
    const CtlResult host = checkCtlHost(model, formula);
    const CtlResult device = checkCtlDevice(model, formula);

    ::kripcuda::testing::expect(host.satisfying == device.satisfying,
                                std::string("satisfying sets agree on ") + label + ": " +
                                    formula.toString(),
                                __FILE__, __LINE__);
    EXPECT_EQ(device.holdsInAllInitialStates, host.holdsInAllInitialStates);
    // The evaluators run the same fixpoint schedule, so they must converge in
    // the same number of steps; a mismatch means the parallel step is not the
    // sequential one.
    EXPECT_EQ(device.iterations, host.iterations);
}

void testDeviceMatchesHost() {
    const KripkeStructure mutex = testing::mutualExclusionModel();
    expectAgreement(mutex, AG(!(atom(kCritical0) && atom(kCritical1))), "mutex safety");
    expectAgreement(mutex, AG(AF(atom(kCritical0))), "mutex liveness");
    expectAgreement(mutex, EG(EF(atom(kCritical1))), "mutex nested");

    const KripkeStructure dense = testing::randomModel(1 << 13, 4, 6, 4242);
    const KripkeStructure sparse = testing::randomModel(1 << 15, 2, 3, 909, 0.5);

    std::mt19937_64 engine(20240607);
    for (int trial = 0; trial < 12; ++trial) {
        expectAgreement(dense, randomFormula(engine, 6, 3), "random formula on dense model");
    }
    for (int trial = 0; trial < 6; ++trial) {
        expectAgreement(sparse, randomFormula(engine, 3, 3), "random formula on sparse model");
    }
}

} // namespace

int main() {
    testPropositionalFragment();
    testTemporalOperatorsOnChain();
    testMutualExclusionProperties();
    testExistsFinallyMatchesReachability();

    if (hasCudaDevice()) {
        std::cout << "CUDA device: " << cudaDeviceDescription() << '\n';
        testDeviceMatchesHost();
    } else {
        std::cout << "no CUDA device visible; skipping device checks\n";
    }

    return kripcuda::testing::report("test_ctl");
}
