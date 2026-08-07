#include "kripcuda/cuda/runtime.hpp"
#include "kripcuda/cuda/stream.cuh"
#include "kripcuda/cuda/transpose.cuh"
#include "kripcuda/verification/fairness_device.cuh"
#include "kripcuda/verification/scc_device.cuh"

#include "models.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

using namespace kripcuda;

namespace {

/// Sequential reference: Tarjan's algorithm, written iteratively so that a
/// state space of a few hundred thousand states does not overflow the host
/// stack. `region` restricts the graph to an induced subgraph; states outside
/// it get no component.
///
/// The reference labels each component by its smallest member, which is not the
/// labelling the device produces — the colouring algorithm names a component
/// after its largest member — so the comparisons below canonicalise both sides
/// before comparing.
std::vector<StateId> tarjanComponents(const KripkeStructure& model,
                                      const std::vector<bool>* region = nullptr) {
    const StateId states = model.stateCount();
    const auto inRegion = [&](StateId state) { return region == nullptr || (*region)[state]; };

    std::vector<StateId> index(states, kInvalidState);
    std::vector<StateId> lowLink(states, 0);
    std::vector<StateId> component(states, kInvalidState);
    std::vector<char> onStack(states, 0);
    std::vector<StateId> stack;
    StateId nextIndex = 0;

    struct Frame {
        StateId state;
        std::size_t edge;
    };
    std::vector<Frame> callStack;

    for (StateId root = 0; root < states; ++root) {
        if (!inRegion(root) || index[root] != kInvalidState) {
            continue;
        }

        callStack.push_back(Frame{root, 0});
        index[root] = lowLink[root] = nextIndex++;
        stack.push_back(root);
        onStack[root] = 1;

        while (!callStack.empty()) {
            Frame& frame = callStack.back();
            const std::span<const StateId> successors = model.successors(frame.state);

            if (frame.edge < successors.size()) {
                const StateId successor = successors[frame.edge++];
                if (!inRegion(successor)) {
                    continue;
                }
                if (index[successor] == kInvalidState) {
                    index[successor] = lowLink[successor] = nextIndex++;
                    stack.push_back(successor);
                    onStack[successor] = 1;
                    callStack.push_back(Frame{successor, 0});
                } else if (onStack[successor] != 0) {
                    lowLink[frame.state] = std::min(lowLink[frame.state], index[successor]);
                }
                continue;
            }

            const StateId finished = frame.state;
            callStack.pop_back();
            if (!callStack.empty()) {
                lowLink[callStack.back().state] =
                    std::min(lowLink[callStack.back().state], lowLink[finished]);
            }

            if (lowLink[finished] == index[finished]) {
                StateId representative = finished;
                for (auto member = stack.rbegin(); *member != finished; ++member) {
                    representative = std::min(representative, *member);
                }
                for (;;) {
                    const StateId member = stack.back();
                    stack.pop_back();
                    onStack[member] = 0;
                    component[member] = representative;
                    if (member == finished) {
                        break;
                    }
                }
            }
        }
    }
    return component;
}

/// Rewrites a partition so that every component is named by its smallest
/// member, which makes two labellings of the same partition equal.
std::vector<StateId> canonicalise(const std::vector<StateId>& component) {
    const auto states = static_cast<StateId>(component.size());
    std::vector<StateId> smallest(component.size(), kInvalidState);
    for (StateId state = 0; state < states; ++state) {
        const StateId representative = component[state];
        if (representative == kInvalidState) {
            continue;
        }
        smallest[representative] = std::min(smallest[representative], state);
    }

    std::vector<StateId> canonical(component.size(), kInvalidState);
    for (StateId state = 0; state < states; ++state) {
        if (component[state] != kInvalidState) {
            canonical[state] = smallest[component[state]];
        }
    }
    return canonical;
}

bool selfLoops(const KripkeStructure& model, StateId state) {
    const std::span<const StateId> successors = model.successors(state);
    return std::find(successors.begin(), successors.end(), state) != successors.end();
}

/// Reference for cyclicStates: the states of a component with more than one
/// member, plus the singletons that loop back to themselves.
std::vector<bool> referenceCyclicStates(const KripkeStructure& model,
                                        const std::vector<StateId>& component) {
    std::vector<StateId> size(model.stateCount(), 0);
    for (const StateId representative : component) {
        if (representative != kInvalidState) {
            ++size[representative];
        }
    }

    std::vector<bool> cyclic(model.stateCount(), false);
    for (StateId state = 0; state < model.stateCount(); ++state) {
        if (component[state] == kInvalidState) {
            continue;
        }
        cyclic[state] = size[component[state]] > 1 || selfLoops(model, state);
    }
    return cyclic;
}

void expectSamePartition(const KripkeStructure& model, const DeviceKripke& device,
                         const DeviceStateSet* region, const std::vector<bool>* hostRegion,
                         const char* what, cudaStream_t stream) {
    const ReverseRelation reverse(device, stream);
    const SccPartition partition = computeSccDevice(device, reverse, region, stream);

    const std::vector<StateId> reference = tarjanComponents(model, hostRegion);
    const std::vector<StateId> expected = canonicalise(reference);
    const std::vector<StateId> actual = canonicalise(partition.componentsToHost(stream));
    ::kripcuda::testing::expect(expected == actual, std::string("SCC partition of ") + what,
                                __FILE__, __LINE__);

    StateId expectedComponents = 0;
    for (StateId state = 0; state < model.stateCount(); ++state) {
        expectedComponents += expected[state] == state ? 1 : 0;
    }
    EXPECT_EQ(partition.statistics().componentCount, expectedComponents);

    const std::vector<bool> expectedCyclic = referenceCyclicStates(model, reference);
    const StateSet cyclic = partition.cyclicStates(device, stream).toHost(stream);
    bool matches = true;
    for (StateId state = 0; state < model.stateCount(); ++state) {
        matches = matches && cyclic.contains(state) == expectedCyclic[state];
    }
    ::kripcuda::testing::expect(matches, std::string("cyclic states of ") + what, __FILE__,
                                __LINE__);
}

void testTranspose() {
    Stream stream;
    const KripkeStructure model = testing::randomModel(4096, 4, 2, 20240607);
    const DeviceKripke device(model, stream);
    const ReverseRelation reverse(device, stream);

    EXPECT_EQ(reverse.transitionCount(), model.transitionCount());

    // The transpose is checked through its row offsets and the multiset of its
    // columns, since the order within a predecessor list is unspecified.
    std::vector<EdgeIndex> offsets(model.stateCount() + 1);
    std::vector<StateId> columns(model.transitionCount());

    std::vector<EdgeIndex> expectedDegrees(model.stateCount(), 0);
    for (const StateId target : model.columns()) {
        ++expectedDegrees[target];
    }

    std::vector<std::vector<StateId>> expectedPredecessors(model.stateCount());
    for (StateId state = 0; state < model.stateCount(); ++state) {
        for (const StateId successor : model.successors(state)) {
            expectedPredecessors[successor].push_back(state);
        }
    }

    // Read the relation back through a kernel-free path: a device state set is
    // not involved, so copy the raw CSR arrays.
    const ReverseRelationView view = reverse.view();
    KRIPCUDA_CHECK(cudaMemcpy(offsets.data(), view.rowOffsets,
                              offsets.size() * sizeof(EdgeIndex), cudaMemcpyDeviceToHost));
    KRIPCUDA_CHECK(cudaMemcpy(columns.data(), view.columns, columns.size() * sizeof(StateId),
                              cudaMemcpyDeviceToHost));

    EXPECT_EQ(offsets.front(), EdgeIndex{0});
    EXPECT_EQ(offsets.back(), static_cast<EdgeIndex>(model.transitionCount()));

    bool matches = true;
    for (StateId state = 0; state < model.stateCount(); ++state) {
        matches = matches && offsets[state + 1] - offsets[state] == expectedDegrees[state];

        std::vector<StateId> actual(columns.begin() + offsets[state],
                                    columns.begin() + offsets[state + 1]);
        std::sort(actual.begin(), actual.end());
        std::sort(expectedPredecessors[state].begin(), expectedPredecessors[state].end());
        matches = matches && actual == expectedPredecessors[state];
    }
    EXPECT_TRUE(matches);
}

void testSccAgainstReference() {
    Stream stream;

    const KripkeStructure mutex = testing::mutualExclusionModel();
    expectSamePartition(mutex, DeviceKripke(mutex, stream), nullptr, nullptr,
                        "the mutual exclusion model", stream);

    // Dense enough that most of the state space collapses into one giant
    // component, which is the case trimming cannot help with.
    const KripkeStructure dense = testing::randomModel(8192, 4, 2, 11);
    expectSamePartition(dense, DeviceKripke(dense, stream), nullptr, nullptr, "a dense model",
                        stream);

    // Sparse and partly unreachable: many singleton components, so trimming
    // decides most of the state space before the colouring runs at all.
    const KripkeStructure sparse = testing::randomModel(8192, 1, 2, 12, 0.25);
    expectSamePartition(sparse, DeviceKripke(sparse, stream), nullptr, nullptr, "a sparse model",
                        stream);
}

void testAcyclicModel() {
    Stream stream;

    // A chain whose last state loops on itself, so the relation stays total.
    constexpr StateId kStates = 1024;
    KripkeBuilder builder(kStates, 1);
    builder.addInitialState(0);
    for (StateId state = 0; state < kStates; ++state) {
        builder.addTransition(state, state + 1 < kStates ? state + 1 : state);
    }
    const KripkeStructure chain = builder.build();

    const DeviceKripke device(chain, stream);
    const SccPartition partition = computeSccDevice(device, stream);

    EXPECT_EQ(partition.statistics().componentCount, kStates);
    EXPECT_EQ(partition.statistics().largestComponent, StateId{1});
    // Only the self-looping final state lies on a cycle.
    EXPECT_EQ(partition.statistics().nontrivialCount, StateId{1});

    const StateSet cyclic = partition.cyclicStates(device, stream).toHost(stream);
    EXPECT_EQ(cyclic.count(), StateId{1});
    EXPECT_TRUE(cyclic.contains(kStates - 1));
}

void testRestrictedRegion() {
    Stream stream;
    const KripkeStructure model = testing::randomModel(4096, 3, 2, 77);
    const DeviceKripke device(model, stream);

    // Every third state, which cuts the giant component into many smaller ones.
    std::vector<bool> hostRegion(model.stateCount(), false);
    StateSet region(model.stateCount());
    for (StateId state = 0; state < model.stateCount(); state += 3) {
        hostRegion[state] = true;
        region.insert(state);
    }

    const DeviceStateSet deviceRegion(region, stream);
    expectSamePartition(model, device, &deviceRegion, &hostRegion, "a restricted region", stream);
}

void testFairCycle() {
    Stream stream;
    const KripkeStructure mutex = testing::mutualExclusionModel();
    const DeviceKripke device(mutex, stream);

    constexpr PropositionId kCritical0 = 0;
    constexpr PropositionId kCritical1 = 1;

    const auto stateSetOf = [&](PropositionId proposition) {
        StateSet set(mutex.stateCount());
        for (StateId state = 0; state < mutex.stateCount(); ++state) {
            if (mutex.holds(state, proposition)) {
                set.insert(state);
            }
        }
        return set;
    };

    // Both processes enter their critical section infinitely often: the model
    // is a single strongly connected component that meets both constraints, so
    // such a path exists.
    FairnessCondition fairness;
    fairness.add(DeviceStateSet(stateSetOf(kCritical0), stream))
        .add(DeviceStateSet(stateSetOf(kCritical1), stream));

    const FairCycleResult fair = checkFairCycle(device, fairness, stream);
    EXPECT_TRUE(fair.holds);
    EXPECT_EQ(fair.witnessCount, mutex.stateCount());

    // An unsatisfiable constraint admits no fair path at all.
    FairnessCondition impossible;
    impossible.add(DeviceStateSet(StateSet(mutex.stateCount()), stream));
    const FairCycleResult none = checkFairCycle(device, impossible, stream);
    EXPECT_TRUE(!none.holds);
    EXPECT_EQ(none.witnessCount, StateId{0});

    // With no constraints every cycle is fair, and every state of this model
    // lies on one.
    const FairCycleResult unconstrained = checkFairCycle(device, FairnessCondition(), stream);
    EXPECT_TRUE(unconstrained.holds);
    EXPECT_EQ(unconstrained.witnessCount, mutex.stateCount());
}

void testFairExistsGlobally() {
    Stream stream;

    // Two disjoint cycles reachable from a common source. Only the right-hand
    // cycle visits the fairness constraint, so EG_fair(true) holds exactly at
    // the source and the right-hand cycle.
    constexpr StateId kSource = 0;
    constexpr StateId kLeftA = 1;
    constexpr StateId kLeftB = 2;
    constexpr StateId kRightA = 3;
    constexpr StateId kRightB = 4;
    constexpr PropositionId kGoal = 0;

    KripkeBuilder builder(5, 1);
    builder.addInitialState(kSource);
    builder.addTransition(kSource, kLeftA).addTransition(kSource, kRightA);
    builder.addTransition(kLeftA, kLeftB).addTransition(kLeftB, kLeftA);
    builder.addTransition(kRightA, kRightB).addTransition(kRightB, kRightA);
    builder.setLabel(kRightA, kGoal);
    const KripkeStructure model = builder.build();

    const DeviceKripke device(model, stream);
    const ReverseRelation reverse(device, stream);

    StateSet goal(model.stateCount());
    goal.insert(kRightA);
    FairnessCondition fairness;
    fairness.add(DeviceStateSet(goal, stream));

    DeviceStateSet everywhere(model.stateCount());
    everywhere.fillAll(stream);

    const StateSet satisfying =
        checkFairExistsGlobally(device, everywhere, fairness, reverse, stream).toHost(stream);
    EXPECT_EQ(satisfying.count(), StateId{3});
    EXPECT_TRUE(satisfying.contains(kSource));
    EXPECT_TRUE(satisfying.contains(kRightA));
    EXPECT_TRUE(satisfying.contains(kRightB));
    EXPECT_TRUE(!satisfying.contains(kLeftA));
    EXPECT_TRUE(!satisfying.contains(kLeftB));

    // Confined to the left-hand cycle there is no fair path anywhere.
    StateSet leftOnly(model.stateCount());
    leftOnly.insert(kLeftA);
    leftOnly.insert(kLeftB);
    const DeviceStateSet leftRegion(leftOnly, stream);
    const StateSet confined =
        checkFairExistsGlobally(device, leftRegion, fairness, reverse, stream).toHost(stream);
    EXPECT_EQ(confined.count(), StateId{0});
}

} // namespace

int main() {
    if (!hasCudaDevice()) {
        std::cout << "no CUDA device visible; skipping SCC test suite\n";
        return kripcuda::testing::report("test_scc");
    }

    testTranspose();
    testSccAgainstReference();
    testAcyclicModel();
    testRestrictedRegion();
    testFairCycle();
    testFairExistsGlobally();
    return kripcuda::testing::report("test_scc");
}
