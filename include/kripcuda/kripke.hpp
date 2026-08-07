#pragma once

#include "kripcuda/types.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kripcuda {

/// Explicit-state Kripke structure M = (S, S0, R, L) over a finite set of
/// atomic propositions. The transition relation is stored in CSR form:
/// the successors of state s are columns[rowOffsets[s] .. rowOffsets[s + 1]).
///
/// Instances are immutable and produced by KripkeBuilder, which guarantees the
/// invariants relied upon by the exploration kernels: successor lists are
/// sorted and duplicate-free, rowOffsets is non-decreasing with
/// rowOffsets[0] == 0 and rowOffsets[|S|] == |R|, and every state has at least
/// one successor (the relation is total, as required for CTL semantics).
class KripkeStructure {
public:
    KripkeStructure() = default;

    [[nodiscard]] StateId stateCount() const noexcept {
        return static_cast<StateId>(row_offsets_.empty() ? 0 : row_offsets_.size() - 1);
    }
    [[nodiscard]] std::size_t transitionCount() const noexcept { return columns_.size(); }
    [[nodiscard]] std::uint32_t propositionCount() const noexcept { return proposition_count_; }
    [[nodiscard]] std::uint32_t labelWordsPerState() const noexcept {
        return labelWordsFor(proposition_count_);
    }

    [[nodiscard]] std::span<const EdgeIndex> rowOffsets() const noexcept { return row_offsets_; }
    [[nodiscard]] std::span<const StateId> columns() const noexcept { return columns_; }
    [[nodiscard]] std::span<const StateId> initialStates() const noexcept { return initial_states_; }
    [[nodiscard]] std::span<const LabelWord> labels() const noexcept { return labels_; }

    [[nodiscard]] std::span<const StateId> successors(StateId state) const {
        const EdgeIndex begin = row_offsets_[state];
        const EdgeIndex end = row_offsets_[state + 1];
        return std::span<const StateId>(columns_).subspan(begin, end - begin);
    }

    [[nodiscard]] bool holds(StateId state, PropositionId proposition) const;

    /// Name of an atomic proposition, or an empty view if none was supplied.
    [[nodiscard]] std::string_view propositionName(PropositionId proposition) const;

private:
    friend class KripkeBuilder;

    std::vector<EdgeIndex> row_offsets_;
    std::vector<StateId> columns_;
    std::vector<StateId> initial_states_;
    std::vector<LabelWord> labels_;
    std::vector<std::string> proposition_names_;
    std::uint32_t proposition_count_ = 0;
};

/// Incremental construction of a KripkeStructure with a fixed state space.
///
/// Transitions may be added in any order and repeated; build() sorts them into
/// CSR layout and removes duplicates so that the resulting structure is
/// independent of insertion order.
class KripkeBuilder {
public:
    KripkeBuilder(StateId stateCount, std::uint32_t propositionCount);

    KripkeBuilder& addTransition(StateId from, StateId to);
    KripkeBuilder& addInitialState(StateId state);
    KripkeBuilder& setLabel(StateId state, PropositionId proposition, bool value = true);
    KripkeBuilder& nameProposition(PropositionId proposition, std::string name);

    /// Throws std::logic_error if the relation is not total or no initial state
    /// was declared. The builder is left empty and must not be reused.
    [[nodiscard]] KripkeStructure build();

private:
    void requireState(StateId state) const;

    StateId state_count_;
    std::uint32_t proposition_count_;
    std::vector<std::pair<StateId, StateId>> edges_;
    std::vector<StateId> initial_states_;
    std::vector<LabelWord> labels_;
    std::vector<std::string> proposition_names_;
};

} // namespace kripcuda
