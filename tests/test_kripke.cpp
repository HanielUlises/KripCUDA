#include "kripcuda/kripke.hpp"

#include "models.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <stdexcept>

using namespace kripcuda;

namespace {

void testCsrLayout() {
    KripkeBuilder builder(3, 0);
    builder.addInitialState(0);
    // Deliberately unsorted and duplicated.
    builder.addTransition(2, 0).addTransition(0, 2).addTransition(0, 1).addTransition(0, 2);
    builder.addTransition(1, 1);
    const KripkeStructure model = builder.build();

    EXPECT_EQ(model.stateCount(), 3u);
    EXPECT_EQ(model.transitionCount(), 4u);
    EXPECT_EQ(model.successors(0).size(), 2u);
    EXPECT_EQ(model.successors(0)[0], 1u);
    EXPECT_EQ(model.successors(0)[1], 2u);
    EXPECT_EQ(model.successors(1)[0], 1u);
    EXPECT_EQ(model.successors(2)[0], 0u);
    EXPECT_EQ(model.rowOffsets()[model.stateCount()], model.transitionCount());
}

void testInsertionOrderIndependence() {
    const auto build = [](bool reversed) {
        KripkeBuilder builder(4, 0);
        builder.addInitialState(0);
        std::vector<std::pair<StateId, StateId>> edges{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 3}};
        if (reversed) {
            std::reverse(edges.begin(), edges.end());
        }
        for (const auto& [from, to] : edges) {
            builder.addTransition(from, to);
        }
        return builder.build();
    };

    const KripkeStructure forward = build(false);
    const KripkeStructure backward = build(true);
    EXPECT_TRUE(std::ranges::equal(forward.rowOffsets(), backward.rowOffsets()));
    EXPECT_TRUE(std::ranges::equal(forward.columns(), backward.columns()));
}

void testLabels() {
    KripkeBuilder builder(2, 70); // spans two label words
    builder.addInitialState(0).addTransition(0, 1).addTransition(1, 0);
    builder.setLabel(0, 0).setLabel(0, 69).setLabel(1, 64).setLabel(1, 64, false);
    builder.nameProposition(69, "high");
    const KripkeStructure model = builder.build();

    EXPECT_EQ(model.labelWordsPerState(), 2u);
    EXPECT_TRUE(model.holds(0, 0));
    EXPECT_TRUE(model.holds(0, 69));
    EXPECT_TRUE(!model.holds(0, 64));
    EXPECT_TRUE(!model.holds(1, 64));
    EXPECT_EQ(model.propositionName(69), std::string_view("high"));
    EXPECT_TRUE(model.propositionName(0).empty());
}

void testValidation() {
    EXPECT_THROWS(KripkeBuilder(0, 0), std::invalid_argument);

    {
        KripkeBuilder builder(2, 1);
        EXPECT_THROWS(builder.addTransition(0, 2), std::out_of_range);
        EXPECT_THROWS(builder.setLabel(0, 1), std::out_of_range);
    }
    {
        KripkeBuilder builder(2, 0);
        builder.addTransition(0, 1).addTransition(1, 0);
        EXPECT_THROWS(static_cast<void>(builder.build()), std::logic_error);
    }
    {
        // State 1 has no successor: the relation is not total.
        KripkeBuilder builder(2, 0);
        builder.addInitialState(0).addTransition(0, 1);
        EXPECT_THROWS(static_cast<void>(builder.build()), std::logic_error);
    }
}

void testMutualExclusionFixture() {
    const KripkeStructure model = testing::mutualExclusionModel();
    EXPECT_EQ(model.stateCount(), 9u);
    for (StateId state = 0; state < model.stateCount(); ++state) {
        EXPECT_TRUE(!model.successors(state).empty());
    }
    EXPECT_TRUE(model.holds(8, 0) && model.holds(8, 1)); // (critical, critical)
}

} // namespace

int main() {
    testCsrLayout();
    testInsertionOrderIndependence();
    testLabels();
    testValidation();
    testMutualExclusionFixture();
    return kripcuda::testing::report("test_kripke");
}
