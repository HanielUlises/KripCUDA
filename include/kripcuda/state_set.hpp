#pragma once

#include "kripcuda/types.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace kripcuda {

/// State sets are packed bitsets over S. The word is 32 bits wide so that one
/// warp covers exactly one word: the device kernels build a word with a single
/// __ballot_sync instead of atomics.
using SetWord = std::uint32_t;

inline constexpr int kSetWordBits = 32;

constexpr std::size_t setWordsFor(StateId stateCount) noexcept {
    return (static_cast<std::size_t>(stateCount) + kSetWordBits - 1) / kSetWordBits;
}

/// Mask of the bits of the last word that correspond to actual states.
constexpr SetWord lastWordMaskFor(StateId stateCount) noexcept {
    const unsigned remainder = stateCount % kSetWordBits;
    return remainder == 0 ? ~SetWord{0} : static_cast<SetWord>((SetWord{1} << remainder) - 1);
}

/// Characteristic function of a subset of the state space.
///
/// Invariant: the padding bits above stateCount() in the last word are zero.
/// Every operation preserves it, which is what makes count() and equality
/// comparison meaningful without a separate normalisation step.
class StateSet {
public:
    StateSet() = default;
    explicit StateSet(StateId stateCount);

    [[nodiscard]] StateId stateCount() const noexcept { return state_count_; }
    [[nodiscard]] std::size_t wordCount() const noexcept { return words_.size(); }

    [[nodiscard]] bool contains(StateId state) const;
    void insert(StateId state);
    void erase(StateId state);

    [[nodiscard]] StateId count() const noexcept;
    [[nodiscard]] bool empty() const noexcept { return count() == 0; }

    [[nodiscard]] std::span<const SetWord> words() const noexcept { return words_; }
    [[nodiscard]] std::span<SetWord> words() noexcept { return words_; }

    StateSet& unionWith(const StateSet& other);
    StateSet& intersectWith(const StateSet& other);
    StateSet& complement();

    friend bool operator==(const StateSet& lhs, const StateSet& rhs) noexcept {
        return lhs.state_count_ == rhs.state_count_ && lhs.words_ == rhs.words_;
    }

private:
    void requireSameDomain(const StateSet& other) const;

    std::vector<SetWord> words_;
    StateId state_count_ = 0;
};

} // namespace kripcuda
