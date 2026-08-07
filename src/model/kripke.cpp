#include "kripcuda/kripke.hpp"

#include <algorithm>
#include <stdexcept>

namespace kripcuda {

bool KripkeStructure::holds(StateId state, PropositionId proposition) const {
    if (state >= stateCount() || proposition >= proposition_count_) {
        throw std::out_of_range("KripkeStructure::holds: index out of range");
    }
    const std::size_t index =
        static_cast<std::size_t>(state) * labelWordsPerState() + proposition / kLabelWordBits;
    const LabelWord mask = LabelWord{1} << (proposition % kLabelWordBits);
    return (labels_[index] & mask) != 0;
}

std::string_view KripkeStructure::propositionName(PropositionId proposition) const {
    if (proposition >= proposition_names_.size()) {
        return {};
    }
    return proposition_names_[proposition];
}

KripkeStructure KripkeStructure::fromValidatedCsr(std::vector<EdgeIndex> rowOffsets,
                                                  std::vector<StateId> columns,
                                                  std::vector<StateId> initialStates,
                                                  std::vector<LabelWord> labels,
                                                  std::uint32_t propositionCount) {
    if (rowOffsets.empty()) {
        throw std::invalid_argument("KripkeStructure::fromValidatedCsr: empty row offsets");
    }
    if (rowOffsets.back() != columns.size()) {
        throw std::invalid_argument(
            "KripkeStructure::fromValidatedCsr: row offsets do not match the column count");
    }

    KripkeStructure structure;
    structure.row_offsets_ = std::move(rowOffsets);
    structure.columns_ = std::move(columns);
    structure.initial_states_ = std::move(initialStates);
    structure.labels_ = std::move(labels);
    structure.proposition_names_.resize(propositionCount);
    structure.proposition_count_ = propositionCount;
    return structure;
}

KripkeBuilder::KripkeBuilder(StateId stateCount, std::uint32_t propositionCount)
    : state_count_(stateCount),
      proposition_count_(propositionCount),
      labels_(static_cast<std::size_t>(stateCount) * labelWordsFor(propositionCount), 0),
      proposition_names_(propositionCount) {
    if (stateCount == 0) {
        throw std::invalid_argument("KripkeBuilder: state space must be non-empty");
    }
}

void KripkeBuilder::requireState(StateId state) const {
    if (state >= state_count_) {
        throw std::out_of_range("KripkeBuilder: state index out of range");
    }
}

KripkeBuilder& KripkeBuilder::addTransition(StateId from, StateId to) {
    requireState(from);
    requireState(to);
    edges_.emplace_back(from, to);
    return *this;
}

KripkeBuilder& KripkeBuilder::addInitialState(StateId state) {
    requireState(state);
    initial_states_.push_back(state);
    return *this;
}

KripkeBuilder& KripkeBuilder::setLabel(StateId state, PropositionId proposition, bool value) {
    requireState(state);
    if (proposition >= proposition_count_) {
        throw std::out_of_range("KripkeBuilder: proposition index out of range");
    }
    const std::size_t index = static_cast<std::size_t>(state) * labelWordsFor(proposition_count_) +
                              proposition / kLabelWordBits;
    const LabelWord mask = LabelWord{1} << (proposition % kLabelWordBits);
    if (value) {
        labels_[index] |= mask;
    } else {
        labels_[index] &= ~mask;
    }
    return *this;
}

KripkeBuilder& KripkeBuilder::nameProposition(PropositionId proposition, std::string name) {
    if (proposition >= proposition_count_) {
        throw std::out_of_range("KripkeBuilder: proposition index out of range");
    }
    proposition_names_[proposition] = std::move(name);
    return *this;
}

KripkeStructure KripkeBuilder::build() {
    if (initial_states_.empty()) {
        throw std::logic_error("KripkeBuilder: no initial state declared");
    }

    std::sort(edges_.begin(), edges_.end());
    edges_.erase(std::unique(edges_.begin(), edges_.end()), edges_.end());

    std::sort(initial_states_.begin(), initial_states_.end());
    initial_states_.erase(std::unique(initial_states_.begin(), initial_states_.end()),
                          initial_states_.end());

    KripkeStructure structure;
    structure.row_offsets_.assign(static_cast<std::size_t>(state_count_) + 1, 0);
    structure.columns_.reserve(edges_.size());

    // edges_ is sorted by source, so a single pass yields the CSR layout.
    for (const auto& [from, to] : edges_) {
        structure.row_offsets_[from + 1] += 1;
        structure.columns_.push_back(to);
    }
    for (StateId state = 0; state < state_count_; ++state) {
        structure.row_offsets_[state + 1] += structure.row_offsets_[state];
    }

    for (StateId state = 0; state < state_count_; ++state) {
        if (structure.row_offsets_[state] == structure.row_offsets_[state + 1]) {
            throw std::logic_error("KripkeBuilder: state " + std::to_string(state) +
                                   " has no successor; the transition relation must be total");
        }
    }

    structure.initial_states_ = std::move(initial_states_);
    structure.labels_ = std::move(labels_);
    structure.proposition_names_ = std::move(proposition_names_);
    structure.proposition_count_ = proposition_count_;

    edges_.clear();
    edges_.shrink_to_fit();
    state_count_ = 0;
    proposition_count_ = 0;

    return structure;
}

} // namespace kripcuda
