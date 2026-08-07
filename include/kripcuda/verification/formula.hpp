#pragma once

#include "kripcuda/types.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace kripcuda::ctl {

/// A CTL state formula, represented as an immutable DAG with value semantics.
///
/// Only an adequate set of operators is primitive — false, atomic
/// propositions, negation, conjunction, disjunction, EX, EU and EG. Every other
/// CTL operator is a derived constructor that rewrites into this basis, so both
/// evaluators implement seven cases and no more. Sharing is preserved: a
/// subformula used twice is one node, and the evaluators memoise on node
/// identity, which is what keeps rewritten operators such as AU affordable.
class Formula {
public:
    enum class Kind : std::uint8_t {
        False,
        Atom,
        Negation,
        Conjunction,
        Disjunction,
        ExistsNext,
        ExistsUntil,
        ExistsGlobally,
    };

    [[nodiscard]] Kind kind() const noexcept { return node_->kind; }

    /// Valid for Kind::Atom.
    [[nodiscard]] PropositionId proposition() const noexcept { return node_->proposition; }

    /// Left operand; for unary operators, the only operand.
    [[nodiscard]] Formula left() const noexcept { return Formula(node_->left); }
    [[nodiscard]] Formula right() const noexcept { return Formula(node_->right); }

    /// Stable identity of the shared node, for memoisation during evaluation.
    [[nodiscard]] const void* identity() const noexcept { return node_.get(); }

    [[nodiscard]] std::string toString() const;

private:
    struct Node {
        Kind kind;
        PropositionId proposition;
        std::shared_ptr<const Node> left;
        std::shared_ptr<const Node> right;
    };

    explicit Formula(std::shared_ptr<const Node> node) noexcept : node_(std::move(node)) {}

    static Formula make(Kind kind, PropositionId proposition, Formula left, Formula right);

    std::shared_ptr<const Node> node_;

    friend Formula constantFalse();
    friend Formula atom(PropositionId);
    friend Formula operator!(Formula);
    friend Formula operator&&(Formula, Formula);
    friend Formula operator||(Formula, Formula);
    friend Formula EX(Formula);
    friend Formula EU(Formula, Formula);
    friend Formula EG(Formula);
};

[[nodiscard]] Formula constantFalse();
[[nodiscard]] Formula constantTrue();
[[nodiscard]] Formula atom(PropositionId proposition);

[[nodiscard]] Formula operator!(Formula operand);
[[nodiscard]] Formula operator&&(Formula left, Formula right);
[[nodiscard]] Formula operator||(Formula left, Formula right);
[[nodiscard]] Formula implies(Formula antecedent, Formula consequent);

[[nodiscard]] Formula EX(Formula operand);
[[nodiscard]] Formula EU(Formula left, Formula right);
[[nodiscard]] Formula EG(Formula operand);

/// Derived operators, by the standard CTL identities:
///   EF ψ    = E[true U ψ]
///   AX φ    = ¬EX ¬φ
///   AF φ    = ¬EG ¬φ
///   AG φ    = ¬EF ¬φ
///   A[φ U ψ] = ¬(E[¬ψ U (¬φ ∧ ¬ψ)] ∨ EG ¬ψ)
[[nodiscard]] Formula EF(Formula operand);
[[nodiscard]] Formula AX(Formula operand);
[[nodiscard]] Formula AF(Formula operand);
[[nodiscard]] Formula AG(Formula operand);
[[nodiscard]] Formula AU(Formula left, Formula right);

} // namespace kripcuda::ctl
