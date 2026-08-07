#include "kripcuda/state_set.hpp"

#include "test_support.hpp"

#include <stdexcept>

using namespace kripcuda;

namespace {

void testBasics() {
    StateSet set(70); // spans three words, last one partial
    EXPECT_EQ(set.wordCount(), 3u);
    EXPECT_EQ(set.count(), 0u);
    EXPECT_TRUE(set.empty());

    set.insert(0);
    set.insert(31);
    set.insert(32);
    set.insert(69);
    EXPECT_EQ(set.count(), 4u);
    EXPECT_TRUE(set.contains(0) && set.contains(31) && set.contains(32) && set.contains(69));
    EXPECT_TRUE(!set.contains(1));

    set.erase(31);
    EXPECT_TRUE(!set.contains(31));
    EXPECT_EQ(set.count(), 3u);

    EXPECT_THROWS(set.insert(70), std::out_of_range);
    EXPECT_THROWS(static_cast<void>(set.contains(70)), std::out_of_range);
}

/// The padding bits above stateCount must stay zero; otherwise complement
/// would invent states and every cardinality would be wrong.
void testComplementMasksPadding() {
    StateSet set(70);
    set.insert(3);
    set.complement();
    EXPECT_EQ(set.count(), 69u);
    EXPECT_TRUE(!set.contains(3));
    EXPECT_TRUE(set.contains(69));

    StateSet aligned(64);
    aligned.complement();
    EXPECT_EQ(aligned.count(), 64u);
}

void testAlgebra() {
    StateSet left(100);
    StateSet right(100);
    for (StateId state = 0; state < 100; state += 2) {
        left.insert(state);
    }
    for (StateId state = 0; state < 100; state += 3) {
        right.insert(state);
    }

    StateSet intersection = left;
    intersection.intersectWith(right);
    EXPECT_EQ(intersection.count(), 17u); // multiples of 6 below 100

    StateSet united = left;
    united.unionWith(right);
    EXPECT_EQ(united.count(), 50u + 34u - 17u);

    StateSet complemented = left;
    complemented.complement();
    complemented.complement();
    EXPECT_TRUE(complemented == left);

    StateSet other(99);
    EXPECT_THROWS(left.unionWith(other), std::invalid_argument);
}

} // namespace

int main() {
    testBasics();
    testComplementMasksPadding();
    testAlgebra();
    return kripcuda::testing::report("test_state_set");
}
