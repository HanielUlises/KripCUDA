#include "kripcuda/state_set.hpp"

#include <bit>
#include <stdexcept>

namespace kripcuda {

StateSet::StateSet(StateId stateCount)
    : words_(setWordsFor(stateCount), 0), state_count_(stateCount) {}

bool StateSet::contains(StateId state) const {
    if (state >= state_count_) {
        throw std::out_of_range("StateSet::contains: state index out of range");
    }
    return (words_[state / kSetWordBits] >> (state % kSetWordBits) & SetWord{1}) != 0;
}

void StateSet::insert(StateId state) {
    if (state >= state_count_) {
        throw std::out_of_range("StateSet::insert: state index out of range");
    }
    words_[state / kSetWordBits] |= SetWord{1} << (state % kSetWordBits);
}

void StateSet::erase(StateId state) {
    if (state >= state_count_) {
        throw std::out_of_range("StateSet::erase: state index out of range");
    }
    words_[state / kSetWordBits] &= ~(SetWord{1} << (state % kSetWordBits));
}

StateId StateSet::count() const noexcept {
    StateId total = 0;
    for (const SetWord word : words_) {
        total += static_cast<StateId>(std::popcount(word));
    }
    return total;
}

void StateSet::requireSameDomain(const StateSet& other) const {
    if (other.state_count_ != state_count_) {
        throw std::invalid_argument("StateSet: operands range over different state spaces");
    }
}

StateSet& StateSet::unionWith(const StateSet& other) {
    requireSameDomain(other);
    for (std::size_t word = 0; word < words_.size(); ++word) {
        words_[word] |= other.words_[word];
    }
    return *this;
}

StateSet& StateSet::intersectWith(const StateSet& other) {
    requireSameDomain(other);
    for (std::size_t word = 0; word < words_.size(); ++word) {
        words_[word] &= other.words_[word];
    }
    return *this;
}

StateSet& StateSet::complement() {
    for (SetWord& word : words_) {
        word = ~word;
    }
    if (!words_.empty()) {
        words_.back() &= lastWordMaskFor(state_count_);
    }
    return *this;
}

} // namespace kripcuda
