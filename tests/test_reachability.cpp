#include "kripcuda/cuda/runtime.hpp"
#include "kripcuda/exploration/reachability.hpp"

#include "models.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <iostream>

using namespace kripcuda;

namespace {

void testHostReachability() {
    // 0 -> 1 -> 2 -> 0 with a self-looping state 3 detached from the cycle.
    KripkeBuilder builder(4, 0);
    builder.addInitialState(0);
    builder.addTransition(0, 1).addTransition(1, 2).addTransition(2, 0).addTransition(3, 3);
    const KripkeStructure model = builder.build();

    const ReachabilityResult result = computeReachabilityHost(model);
    EXPECT_EQ(result.levels[0], 0);
    EXPECT_EQ(result.levels[1], 1);
    EXPECT_EQ(result.levels[2], 2);
    EXPECT_EQ(result.levels[3], kUnreachable);
    EXPECT_EQ(result.reachableCount, 3u);
    EXPECT_EQ(result.maxLevel, 2);
    EXPECT_TRUE(!result.isReachable(3));
}

void testMultipleInitialStates() {
    KripkeBuilder builder(5, 0);
    builder.addInitialState(0).addInitialState(4);
    builder.addTransition(0, 1).addTransition(1, 2).addTransition(2, 3);
    builder.addTransition(3, 3).addTransition(4, 3);
    const KripkeStructure model = builder.build();

    const ReachabilityResult result = computeReachabilityHost(model);
    EXPECT_EQ(result.levels[4], 0);
    EXPECT_EQ(result.levels[3], 1); // reached from state 4, not through the chain
    EXPECT_EQ(result.reachableCount, 5u);
}

void testMutualExclusionIsSafe() {
    const KripkeStructure model = testing::mutualExclusionModel();
    const ReachabilityResult result = computeReachabilityHost(model);

    StateId unsafeReachable = 0;
    for (StateId state = 0; state < model.stateCount(); ++state) {
        if (result.isReachable(state) && model.holds(state, 0) && model.holds(state, 1)) {
            ++unsafeReachable;
        }
    }
    EXPECT_EQ(unsafeReachable, 0u);
    EXPECT_EQ(result.reachableCount, 8u); // every state but (critical, critical)
}

void expectAgreement(const KripkeStructure& model, const char* label) {
    const ReachabilityResult host = computeReachabilityHost(model);
    const ReachabilityResult device = computeReachabilityDevice(model);

    const bool levelsMatch = std::ranges::equal(host.levels, device.levels);
    ::kripcuda::testing::expect(levelsMatch, std::string("levels agree on ") + label, __FILE__,
                                __LINE__);
    EXPECT_EQ(device.reachableCount, host.reachableCount);
    EXPECT_EQ(device.maxLevel, host.maxLevel);
}

void testDeviceMatchesHost() {
    expectAgreement(testing::mutualExclusionModel(), "mutual exclusion");
    expectAgreement(testing::randomModel(1 << 14, 4, 8, 1234), "dense random model");
    expectAgreement(testing::randomModel(1 << 16, 3, 64, 99, 0.5), "half-unreachable model");

    // Long, thin state space: exercises many BFS levels with a tiny frontier.
    KripkeBuilder chain(1 << 12, 0);
    chain.addInitialState(0);
    for (StateId state = 0; state + 1 < (1 << 12); ++state) {
        chain.addTransition(state, state + 1);
    }
    chain.addTransition((1 << 12) - 1, 0);
    expectAgreement(chain.build(), "chain");
}

} // namespace

int main() {
    testHostReachability();
    testMultipleInitialStates();
    testMutualExclusionIsSafe();

    if (hasCudaDevice()) {
        std::cout << "CUDA device: " << cudaDeviceDescription() << '\n';
        testDeviceMatchesHost();
    } else {
        std::cout << "no CUDA device visible; skipping device checks\n";
    }

    return kripcuda::testing::report("test_reachability");
}
